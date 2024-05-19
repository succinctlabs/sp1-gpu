use crate::device::buffer::DeviceBuffer;
use crate::device::buffer::ToDevice;

use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;

#[derive(Debug)]
#[repr(C)]
pub struct RowMajorMatrixDevice<T: Copy> {
    pub values: DeviceBuffer<T>,
    pub width: usize,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct RowMajorMatrixViewDevice<T> {
    pub values: *const T,
    pub width: usize,
    pub height: usize,
}

impl<T: Copy + Send + Sync> RowMajorMatrixDevice<T> {
    pub fn rand(width: usize, height: usize) -> Self
    where
        rand::distributions::Standard: rand::distributions::Distribution<T>,
    {
        let mut rng = rand::thread_rng();
        let data = (0..width * height).map(|_| rng.gen()).collect::<Vec<_>>();
        RowMajorMatrixDevice {
            values: data.to_device(),
            width,
        }
    }

    pub fn view(&self) -> RowMajorMatrixViewDevice<T> {
        RowMajorMatrixViewDevice {
            values: self.values.as_ptr(),
            width: self.width,
            height: self.values.len() / self.width,
        }
    }

    pub fn to_host(&self) -> RowMajorMatrix<T> {
        RowMajorMatrix::new(self.values.to_host(), self.width)
    }
}

pub mod merkle_tree_gpu {
    use p3_baby_bear::BabyBear;

    use crate::merkle_tree::RowMajorMatrixViewDevice;
    use crate::poseidon2::poseidon2_bb31_16_gpu::{DIGEST_WIDTH, WIDTH};

    #[allow(unused_attributes)]
    #[link_name = "merkle_tree_gpu"]
    extern "C" {
        #[link_name = "firstDigestLayer"]
        pub fn first_digest_layer(
            tallest_matrices: *const RowMajorMatrixViewDevice<BabyBear>,
            n_tallest_matrices: usize,
            digests: *mut [BabyBear; DIGEST_WIDTH],
            n_blocks: usize,
            n_threads_per_block: usize,
        );

        #[link_name = "compressAndInject"]
        pub fn compress_and_inject(
            prev_layer: *const [BabyBear; WIDTH],
            n_prev_layer: usize,
            matrices_to_inject: *const RowMajorMatrixViewDevice<BabyBear>,
            n_matrices_to_inject: usize,
            next_digests: *mut [BabyBear; DIGEST_WIDTH],
            n_blocks: usize,
            n_threads_per_block: usize,
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::poseidon2::tests::poseidon2_bb31_16_hasher;
    use crate::{
        device::buffer::{DeviceBuffer, ToDevice},
        merkle_tree::merkle_tree_gpu,
        poseidon2::poseidon2_bb31_16_gpu::DIGEST_WIDTH,
    };

    use super::RowMajorMatrixDevice;
    use p3_baby_bear::BabyBear;

    #[test]
    fn test_first_digest_layer() {
        let n = 1 << 5;
        let tallest_matrices = [
            RowMajorMatrixDevice::<BabyBear>::rand(9, n),
            RowMajorMatrixDevice::<BabyBear>::rand(4, n),
        ];
        let tallest_matrices_view = tallest_matrices
            .iter()
            .map(|m| m.view())
            .collect::<Vec<_>>()
            .to_device();
        let mut digests = DeviceBuffer::<[BabyBear; DIGEST_WIDTH]>::with_capacity(n);

        let start = std::time::Instant::now();
        unsafe {
            digests.set_len(n);
            merkle_tree_gpu::first_digest_layer(
                tallest_matrices_view.as_ptr(),
                tallest_matrices_view.len(),
                digests.as_mut_ptr(),
                n / 32,
                32,
            );
        }
        println!("{:?}", start.elapsed().as_secs_f64());

        let tallest_matrices = tallest_matrices
            .iter()
            .map(|m| m.to_host())
            .collect::<Vec<_>>();
        let tallest_matrices_ref = tallest_matrices.iter().collect::<Vec<_>>();
        let hasher = poseidon2_bb31_16_hasher();
        let digests_host = p3_merkle_tree::first_digest_layer(&hasher, tallest_matrices_ref);
        let digests_device = digests.to_host();

        for i in 0..n {
            assert_eq!(digests_host[i], digests_device[i]);
        }
    }
}
