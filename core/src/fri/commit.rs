use std::borrow::Borrow;

use p3_baby_bear::BabyBear;
use p3_commit::{PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_field::{AbstractField, Field};
use sp1_core::stark::Com;

use crate::cuda_runtime::stream::CudaStream;
use crate::device::error::CudaError;
use crate::device::RawDevicePointer;
use crate::dft::{DeviceDft, Dft};
use crate::matrix::{ColMajorMatrix, ColMajorMatrixDevice};
use crate::merkle_tree::MmcsCommitter;
use crate::stark::BabyBearFriConfig;

pub struct TwoAdicFriCommitter<SC: BabyBearFriConfig, C> {
    pub dft: DeviceDft<SC::Val>,
    pub mmcs_committer: C,
    pub log_blowup: usize,
}

impl<
        SC: BabyBearFriConfig,
        C: MmcsCommitter<BabyBear, SC::ValMmcs, Matrix = ColMajorMatrixDevice<SC::Val>>,
    > TwoAdicFriCommitter<SC, C>
{
    pub fn new(log_blowup: usize) -> Self
    where
        C: Default,
    {
        Self {
            dft: DeviceDft::new(),
            mmcs_committer: C::default(),
            log_blowup,
        }
    }

    pub fn mmcs_commit(&self, leaves: Vec<C::Matrix>) -> (Com<SC>, C::ProverData) {
        self.mmcs_committer.commit(leaves)
    }

    pub const fn log_blowup(&self) -> usize {
        self.log_blowup
    }

    pub fn encode<P>(
        &self,
        domain: TwoAdicMultiplicativeCoset<BabyBear>,
        matrix: &ColMajorMatrix<P>,
        bit_reversed: bool,
    ) -> Result<ColMajorMatrix<P>, CudaError>
    where
        P: RawDevicePointer<Data = BabyBear>,
        DeviceDft<BabyBear>: Dft<P>,
    {
        assert_eq!(domain.size(), matrix.height());

        let shift = domain.shift.inverse();
        unsafe {
            let mut lde_mat = matrix.embed_as_blowup(self.log_blowup)?;
            self.dft
                .coset_lde_batch(&mut lde_mat, self.log_blowup, shift, bit_reversed)?;

            Ok(lde_mat)
        }
    }

    pub fn get_evaluations_on_domain(
        &self,
        src_domain: TwoAdicMultiplicativeCoset<BabyBear>,
        dst_domain: TwoAdicMultiplicativeCoset<BabyBear>,
        matrix: &ColMajorMatrixDevice<BabyBear>,
    ) -> Result<ColMajorMatrixDevice<BabyBear>, CudaError> {
        // Domain assertions for the current usage. The code is supposed to work regardless but we
        // keep them here for now since other usages are untested.
        debug_assert!(src_domain.shift.is_one());
        debug_assert_eq!(dst_domain.shift, BabyBear::generator());

        let shift = dst_domain.shift * BabyBear::generator().inverse() * src_domain.shift.inverse();
        let log_blowup = dst_domain.log_n - src_domain.log_n;
        unsafe {
            let mut lde_mat = matrix.embed_as_blowup(log_blowup)?;
            self.dft
                .coset_lde_batch(&mut lde_mat, log_blowup, shift, false)?;

            Ok(lde_mat)
        }
    }

    pub fn commit<M>(
        &self,
        evaluations: &[(TwoAdicMultiplicativeCoset<BabyBear>, M)],
    ) -> (Com<SC>, C::ProverData)
    where
        M: Borrow<C::Matrix>,
    {
        let lde_evaluations = evaluations
            .iter()
            .map(|(domain, matrix)| {
                let matrix = matrix.borrow();
                self.encode(*domain, matrix, true).unwrap()
            })
            .collect::<Vec<_>>();

        self.mmcs_commit(lde_evaluations)
    }
}

#[cfg(test)]
mod tests {
    use crate::{device::memory::ToDevice, time::CudaInstant};

    use super::*;
    use p3_commit::Pcs;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::thread_rng;

    use p3_field::AbstractField;

    use sp1_core::{stark::StarkGenericConfig, utils::BabyBearPoseidon2};

    use crate::merkle_tree::Poseidon2BabyBearCommitter;

    #[test]
    fn test_commit_device() {
        let log_blowup = 1;
        let log_degrees = [16, 10, 8];
        let columns = [100, 200, 300];

        type SC = BabyBearPoseidon2;

        let mut rng = thread_rng();

        let domains_and_traces = log_degrees
            .iter()
            .zip(columns.iter())
            .map(|(log_degree, cols)| {
                let trace = RowMajorMatrix::<BabyBear>::rand(&mut rng, 1 << log_degree, *cols);

                let domain = TwoAdicMultiplicativeCoset::<BabyBear> {
                    log_n: *log_degree,
                    shift: BabyBear::one(),
                };

                (domain, trace)
            })
            .collect::<Vec<_>>();

        let evaluations = domains_and_traces
            .iter()
            .map(|(domain, trace)| {
                let trace = trace.to_device().unwrap();
                (*domain, trace)
            })
            .collect::<Vec<_>>();

        let pcs = TwoAdicFriCommitter::<SC, Poseidon2BabyBearCommitter>::new(log_blowup);
        let time = CudaInstant::now().unwrap();
        let (commit, _) = pcs.commit(&evaluations);
        println!("time: {:?}", time.elapsed().unwrap());

        let sp1_config = SC::default();
        let (expected_commit, _) = <<SC as StarkGenericConfig>::Pcs as Pcs<
            <SC as StarkGenericConfig>::Challenge,
            <SC as StarkGenericConfig>::Challenger,
        >>::commit(sp1_config.pcs(), domains_and_traces);

        assert_eq!(commit, expected_commit);
    }
}
