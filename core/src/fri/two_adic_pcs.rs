use std::time::Instant;

use p3_baby_bear::BabyBear;
use p3_commit::{PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use crate::device::buffer::DeviceBuffer;
use crate::dft::DeviceDft;
use crate::matrix::ColMajorMatrixDevice;
use crate::merkle_tree::FieldMerkleTreeGpu;
use crate::poseidon2::poseidon2_bb31_16_kernels::DIGEST_WIDTH;

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
                let d = domain.size();
                let ext_d = matrix.height() << self.log_blowup;

                let mut lde_buffer = DeviceBuffer::<BabyBear>::with_capacity(
                    (matrix.height() * matrix.width()) << self.log_blowup,
                );

                unsafe { lde_buffer.set_max_len() };
                let mut lde_mat = ColMajorMatrixDevice::new(lde_buffer, ext_d);

                let matrix = matrix.transpose();

                for j in 0..matrix.height() {
                    let col = matrix.row_slice(j);
                    lde_mat.values[j * ext_d + ext_d - d..(j + 1) * ext_d].copy_from_host(&col);
                }

                unsafe {
                    self.dft
                        .coset_lde_batch_device(lde_mat.view_mut(), self.log_blowup)
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
    use rand::thread_rng;

    use std::time::Instant;

    use p3_commit::Pcs;
    use p3_field::AbstractField;

    use sp1_core::stark::StarkGenericConfig;
    use sp1_core::utils::BabyBearPoseidon2;

    #[test]
    fn test_commit_from_host() {
        let log_blowup = 2;
        let log_degree = 22;
        let degree = 1 << log_degree;
        let n_cols = 600;

        type SC = BabyBearPoseidon2;

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

        let sp1_config = SC::default();
        let (_, _) = <<SC as StarkGenericConfig>::Pcs as Pcs<
            <SC as StarkGenericConfig>::Challenge,
            <SC as StarkGenericConfig>::Challenger,
        >>::commit(sp1_config.pcs(), evaluations_clone);
    }
}
