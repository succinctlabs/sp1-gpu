use std::borrow::Borrow;
use std::thread;

use p3_baby_bear::BabyBear;
use p3_commit::{PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_field::Field;
use p3_symmetric::Hash;

use crate::device::error::CudaError;
use crate::device::CudaSync;
use crate::dft::DeviceDft;
use crate::matrix::ColMajorMatrixDevice;
use crate::merkle_tree::FieldMerkleTreeGpu;
use crate::poseidon2::poseidon2_bb31_16_kernels::DIGEST_WIDTH;

pub struct TwoAdicFriCommitter<F, D, M = CudaSync<ColMajorMatrixDevice<F>>> {
    dft: DeviceDft<F>,
    log_blowup: usize,
    _marker: std::marker::PhantomData<(D, M)>,
}

impl TwoAdicFriCommitter<BabyBear, [BabyBear; DIGEST_WIDTH]> {
    pub fn new(log_blowup: usize) -> Self {
        Self {
            dft: DeviceDft::new(),
            log_blowup,
            _marker: std::marker::PhantomData,
        }
    }

    pub const fn log_blowup(&self) -> usize {
        self.log_blowup
    }

    pub fn encode(
        &self,
        domain: TwoAdicMultiplicativeCoset<BabyBear>,
        matrix: &ColMajorMatrixDevice<BabyBear>,
        bit_reversed: bool,
    ) -> Result<ColMajorMatrixDevice<BabyBear>, CudaError> {
        assert_eq!(domain.size(), matrix.height());

        let shift = domain.shift.inverse();
        unsafe {
            let mut lde_mat = matrix.embed_as_blowup(self.log_blowup)?;
            self.dft.coset_lde_batch_device(
                lde_mat.view_mut(),
                self.log_blowup,
                shift,
                bit_reversed,
            )?;

            Ok(lde_mat)
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn commit<Matrix>(
        &self,
        evaluations: &[(TwoAdicMultiplicativeCoset<BabyBear>, Matrix)],
    ) -> (
        Hash<BabyBear, BabyBear, DIGEST_WIDTH>,
        FieldMerkleTreeGpu<
            BabyBear,
            [BabyBear; DIGEST_WIDTH],
            CudaSync<ColMajorMatrixDevice<BabyBear>>,
        >,
    )
    where
        Matrix: Send + Sync + Borrow<CudaSync<ColMajorMatrixDevice<BabyBear>>>,
    {
        let lde_evaluations = evaluations
            .iter()
            .map(|(domain, matrix)| {
                let matrix = matrix.borrow();
                CudaSync::new(self.encode(*domain, matrix, true).unwrap()).unwrap()
            })
            .collect::<Vec<_>>();

        let tree_device = FieldMerkleTreeGpu::new(lde_evaluations);
        let root_device = tree_device.root().into();

        (root_device, tree_device)
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
                let trace = trace.to_device().to_column_major();
                let trace = CudaSync::new(trace).unwrap();
                (*domain, trace)
            })
            .collect::<Vec<_>>();

        let pcs = TwoAdicFriCommitter::new(log_blowup);
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
