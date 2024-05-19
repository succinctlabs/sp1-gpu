use crate::device::buffer::DeviceBuffer;
use crate::device::buffer::ToDevice;
use crate::poseidon2::poseidon2_bb31_16_kernels::DIGEST_WIDTH;

use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;
use rand::Rng;
use std::cmp::Reverse;

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

    pub fn height(&self) -> usize {
        self.values.len() / self.width
    }
}

pub fn commit(leaves: Vec<RowMajorMatrixViewDevice<BabyBear>>) -> [BabyBear; DIGEST_WIDTH] {
    let mut leaves_largest_first = leaves
        .iter()
        .sorted_by_key(|l| Reverse(l.height))
        .peekable();

    let max_height = leaves_largest_first.peek().unwrap().height;
    let tallest_matrices = leaves_largest_first
        .peeking_take_while(|m| m.height == max_height)
        .copied()
        .collect_vec()
        .to_device();

    let mut first_digest_layer = DeviceBuffer::with_capacity(max_height);
    unsafe {
        first_digest_layer.set_len(max_height);
        merkle_tree_gpu::first_digest_layer(
            tallest_matrices.as_ptr(),
            tallest_matrices.len(),
            first_digest_layer.as_mut_ptr(),
            max_height / 32 + 1,
            32,
        );
    }

    let mut digest_layers = vec![first_digest_layer];
    loop {
        let prev_layer = digest_layers.last().unwrap();
        if prev_layer.len() == 1 {
            break;
        }
        let next_layer_len = prev_layer.len() / 2;

        let matrices_to_inject = leaves_largest_first
            .peeking_take_while(|m| m.height.next_power_of_two() == next_layer_len)
            .copied()
            .collect_vec()
            .to_device();

        let mut next_digests =
            DeviceBuffer::<[BabyBear; DIGEST_WIDTH]>::with_capacity(next_layer_len);
        unsafe {
            next_digests.set_len(next_layer_len);
            merkle_tree_gpu::compress_and_inject(
                prev_layer.as_ptr(),
                prev_layer.len(),
                matrices_to_inject.as_ptr(),
                matrices_to_inject.len(),
                next_digests.as_mut_ptr(),
                next_layer_len / 32 + 1,
                32,
            );
        }
        digest_layers.push(next_digests);
    }

    digest_layers.last().unwrap().to_host()[0]
}

pub mod merkle_tree_gpu {
    use p3_baby_bear::BabyBear;

    use crate::merkle_tree::RowMajorMatrixViewDevice;
    use crate::poseidon2::poseidon2_bb31_16_kernels::DIGEST_WIDTH;

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
            prev_layer: *const [BabyBear; DIGEST_WIDTH],
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
    use crate::poseidon2::tests::{poseidon2_bb31_16_compressor, poseidon2_bb31_16_hasher};
    use crate::{
        device::buffer::{DeviceBuffer, ToDevice},
        merkle_tree::merkle_tree_gpu,
        poseidon2::poseidon2_bb31_16_kernels::DIGEST_WIDTH,
    };

    use super::{commit, RowMajorMatrixDevice};
    use p3_baby_bear::BabyBear;

    #[test]
    fn test_first_digest_layer() {
        let n = 1 << 16;
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

    #[test]
    fn test_compress_and_inject() {
        let n = 1 << 16;

        // Compute first digest layer on GPU.
        let tallest_matrices = [RowMajorMatrixDevice::<BabyBear>::rand(9, n)];
        let tallest_matrices_view = tallest_matrices
            .iter()
            .map(|m| m.view())
            .collect::<Vec<_>>()
            .to_device();
        let mut first_layer_digests = DeviceBuffer::<[BabyBear; DIGEST_WIDTH]>::with_capacity(n);
        unsafe {
            first_layer_digests.set_len(n);
            merkle_tree_gpu::first_digest_layer(
                tallest_matrices_view.as_ptr(),
                tallest_matrices_view.len(),
                first_layer_digests.as_mut_ptr(),
                n / 32,
                32,
            );
        }

        // Compute second layer on GPU.
        let matrices_to_inject = [RowMajorMatrixDevice::<BabyBear>::rand(14, n >> 1)];
        let matrices_to_inject_view = matrices_to_inject
            .iter()
            .map(|m| m.view())
            .collect::<Vec<_>>()
            .to_device();
        let mut next_digests = DeviceBuffer::<[BabyBear; DIGEST_WIDTH]>::with_capacity(n >> 1);
        unsafe {
            next_digests.set_len(n / 2);
            merkle_tree_gpu::compress_and_inject(
                first_layer_digests.as_ptr(),
                first_layer_digests.len(),
                matrices_to_inject_view.as_ptr(),
                matrices_to_inject_view.len(),
                next_digests.as_mut_ptr(),
                n / 32,
                32,
            );
        }

        // Compare the output of the first layer between CPU and GPU.
        let tallest_matrices = tallest_matrices
            .iter()
            .map(|m| m.to_host())
            .collect::<Vec<_>>();
        let tallest_matrices_ref = tallest_matrices.iter().collect::<Vec<_>>();
        let hasher = poseidon2_bb31_16_hasher();
        let first_layer_digests_host =
            p3_merkle_tree::first_digest_layer(&hasher, tallest_matrices_ref);
        let first_layer_digests_device = first_layer_digests.to_host();
        for i in 0..n {
            assert_eq!(first_layer_digests_host[i], first_layer_digests_device[i]);
        }

        // Compare output of the second layer between CPU and GPU.
        let matrices_to_inject = matrices_to_inject
            .iter()
            .map(|m| m.to_host())
            .collect::<Vec<_>>();
        let matrices_to_inject_ref = matrices_to_inject.iter().collect::<Vec<_>>();
        let compressor = poseidon2_bb31_16_compressor();
        let next_digests_host = p3_merkle_tree::compress_and_inject(
            &first_layer_digests_host,
            matrices_to_inject_ref,
            &hasher,
            &compressor,
        );
        let next_digests_device = next_digests.to_host();
        for i in 0..n / 2 {
            assert_eq!(next_digests_host[i], next_digests_device[i]);
        }
    }

    #[test]
    fn test_commit_matrices() {
        let n = 1 << 16;

        let tallest_matrices = [RowMajorMatrixDevice::<BabyBear>::rand(600, n)];
        let tallest_matrices_view = tallest_matrices
            .iter()
            .map(|m| m.view())
            .collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let digest = commit(tallest_matrices_view);

        let tallest_matrices = vec![tallest_matrices[0].to_host()];
        let hasher = poseidon2_bb31_16_hasher();
        let compressor = poseidon2_bb31_16_compressor();
        let tree = p3_merkle_tree::FieldMerkleTree::new(&hasher, &compressor, tallest_matrices);

        let root: [BabyBear; DIGEST_WIDTH] = tree.root().into();
        assert_eq!(digest, root);
        println!("{:?}", start.elapsed().as_secs_f64());
    }
}
