use std::borrow::Borrow;

use p3_baby_bear::BabyBear;
use p3_commit::{PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_field::{AbstractField, Field};
use p3_symmetric::Hash;

use crate::device::error::CudaError;
use crate::dft::DeviceDft;
use crate::matrix::{ColMajorMatrixDevice, DeviceMatrix};
use crate::merkle_tree::{FieldMerkleTreeGpu, FieldMerkleTreeHasher};
use crate::poseidon2::baby_bear::poseidon2_baby_bear_16_kernels::DIGEST_WIDTH;
use crate::poseidon2::baby_bear::DeviceHasherBabyBear;

pub struct TwoAdicFriCommitter<F, H = DeviceHasherBabyBear, M = ColMajorMatrixDevice<F>> {
    dft: DeviceDft<F>,
    hasher: H,
    pub log_blowup: usize,
    _marker: std::marker::PhantomData<(H, M)>,
}

impl<H: FieldMerkleTreeHasher<BabyBear>> TwoAdicFriCommitter<BabyBear, H> {
    pub fn new(log_blowup: usize) -> Self
    where
        H: Default,
    {
        Self {
            dft: DeviceDft::new(),
            hasher: H::default(),
            log_blowup,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn mmcs_commit<Matrix: DeviceMatrix<BabyBear>>(
        &self,
        leaves: Vec<Matrix>,
    ) -> FieldMerkleTreeGpu<BabyBear, H::Digest, Matrix> {
        FieldMerkleTreeGpu::new(&self.hasher, leaves)
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
            let mut lde_mat = matrix.embed_as_blowup(self.log_blowup)?;
            self.dft
                .coset_lde_batch_device(lde_mat.view_mut(), log_blowup, shift, false)?;

            Ok(lde_mat)
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn commit<Matrix>(
        &self,
        evaluations: &[(TwoAdicMultiplicativeCoset<BabyBear>, Matrix)],
    ) -> (
        Hash<BabyBear, BabyBear, DIGEST_WIDTH>,
        FieldMerkleTreeGpu<BabyBear, H::Digest, ColMajorMatrixDevice<BabyBear>>,
    )
    where
        Matrix: Borrow<ColMajorMatrixDevice<BabyBear>>,
        H: FieldMerkleTreeHasher<BabyBear, Digest = [BabyBear; DIGEST_WIDTH]>,
    {
        let lde_evaluations = evaluations
            .iter()
            .map(|(domain, matrix)| {
                let matrix = matrix.borrow();
                self.encode(*domain, matrix, true).unwrap()
            })
            .collect::<Vec<_>>();

        let tree_device = self.mmcs_commit(lde_evaluations);
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
                (*domain, trace)
            })
            .collect::<Vec<_>>();

        let pcs = TwoAdicFriCommitter::<_, DeviceHasherBabyBear, _>::new(log_blowup);
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
