use std::time::Instant;

use p3_baby_bear::BabyBear;
use p3_commit::{PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use crate::dft::DeviceDft;
use crate::matrix::{ColMajorMatrixDevice, RowMajorMatrixDevice};
use crate::merkle_tree::FieldMerkleTreeGpu;
use crate::poseidon2::poseidon2_bb31_16_kernels::DIGEST_WIDTH;

use crate::device::memory::ToDevice;

pub struct TwoAdicFriPcs<F, D, M = ColMajorMatrixDevice<F>> {
    dft: DeviceDft<F>,
    log_blowup: usize,
    _marker: std::marker::PhantomData<(D, M)>,
}

impl TwoAdicFriPcs<BabyBear, [BabyBear; DIGEST_WIDTH]> {
    pub fn new(log_blowup: usize) -> Self {
        Self {
            dft: DeviceDft::new(),
            log_blowup,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn commit_from_host(
        &self,
        evaluations: Vec<(
            TwoAdicMultiplicativeCoset<BabyBear>,
            RowMajorMatrix<BabyBear>,
        )>,
    ) -> (
        [BabyBear; DIGEST_WIDTH],
        FieldMerkleTreeGpu<BabyBear, [BabyBear; DIGEST_WIDTH], ColMajorMatrixDevice<BabyBear>>,
    ) {
        let lde_evaluations = evaluations
            .into_iter()
            .map(|(domain, matrix)| {
                assert_eq!(domain.size(), matrix.height());

                let matrix = RowMajorMatrixDevice::new(matrix.values.to_device(), matrix.width());
                let mut lde_mat = matrix.to_column_major_blowup(self.log_blowup);

                unsafe {
                    self.dft
                        .coset_lde_batch_device(lde_mat.view_mut(), self.log_blowup, true)
                }
                .unwrap();

                lde_mat
            })
            .collect::<Vec<_>>();

        let start = Instant::now();
        let tree_device = FieldMerkleTreeGpu::new(lde_evaluations);
        let root_device = tree_device.root();
        println!("commit time {:?}", start.elapsed());

        (root_device, tree_device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_commit::Pcs;
    use p3_matrix::bitrev::BitReversableMatrix;
    use rand::thread_rng;

    use std::time::Instant;

    use p3_field::AbstractField;

    use sp1_core::{stark::StarkGenericConfig, utils::BabyBearPoseidon2};

    #[test]
    fn test_commit_from_host() {
        let log_blowup = 1;
        let log_degrees = [16];
        let columns = [100];

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

        let evaluations_clone = domains_and_traces.clone();

        let pcs = TwoAdicFriPcs::new(log_blowup);
        let time = Instant::now();
        let (commit, _) = pcs.commit_from_host(domains_and_traces);
        println!("time: {:?}", time.elapsed());

        let sp1_config = SC::default();
        let (expected_commit, _) = <<SC as StarkGenericConfig>::Pcs as Pcs<
            <SC as StarkGenericConfig>::Challenge,
            <SC as StarkGenericConfig>::Challenger,
        >>::commit(sp1_config.pcs(), evaluations_clone);

        let expected_commit: [BabyBear; DIGEST_WIDTH] = expected_commit.into();

        assert_eq!(commit, expected_commit);
    }
}
