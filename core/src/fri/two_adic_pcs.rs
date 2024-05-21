use p3_baby_bear::BabyBear;
use p3_commit::{PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use crate::dft::DeviceDft;
use crate::matrix::{ColMajorMatrixDevice, RowMajorMatrixDevice};
use crate::merkle_tree::FieldMerkleTreeGpu;
use crate::poseidon2::poseidon2_bb31_16_kernels::DIGEST_WIDTH;

use crate::device::buffer::ToDevice;

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
                        .coset_lde_batch_device(lde_mat.view_mut(), self.log_blowup)
                }
                .unwrap();

                lde_mat
            })
            .collect::<Vec<_>>();

        let tree_device = FieldMerkleTreeGpu::new(lde_evaluations);
        let root_device = tree_device.root();

        (root_device, tree_device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    use std::time::Instant;

    use p3_field::AbstractField;

    use sp1_core::utils::BabyBearPoseidon2;

    #[test]
    fn test_commit_from_host() {
        let log_blowup = 1;
        let log_degree = 22;
        let degree = 1 << log_degree;
        let n_cols = 100;

        let mut rng = thread_rng();
        let trace = RowMajorMatrix::<BabyBear>::rand(&mut rng, degree, n_cols);

        let domain = TwoAdicMultiplicativeCoset::<BabyBear> {
            log_n: log_degree,
            shift: BabyBear::one(),
        };

        let evaluations = vec![(domain, trace)];
        let evaluations_clone = evaluations.clone();

        let pcs = TwoAdicFriPcs::new(log_blowup);
        let time = Instant::now();
        let (_, _) = pcs.commit_from_host(evaluations);
        println!("time: {:?}", time.elapsed());

        // let sp1_config = SC::default();
        // let (_, _) = <<SC as StarkGenericConfig>::Pcs as Pcs<
        //     <SC as StarkGenericConfig>::Challenge,
        //     <SC as StarkGenericConfig>::Challenger,
        // >>::commit(sp1_config.pcs(), evaluations_clone);
    }
}
