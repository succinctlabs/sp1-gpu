use crate::device::error::CudaError;
use crate::device::memory::ToDevice;
use crate::device::memory::ToHost;
use crate::device::DeviceBuffer;
use crate::matrix;
use crate::matrix::ColMajorMatrixDevice;

use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::FieldMerkleTree;
use p3_util::log2_ceil_usize;
use std::cmp::Reverse;
use std::marker::PhantomData;

mod hasher;
mod mmcs;
pub use hasher::*;
pub use mmcs::*;

use crate::matrix::DeviceMatrix;
pub struct FieldMerkleTreeGpu<F: Copy, D: Copy, M: DeviceMatrix<F> = ColMajorMatrixDevice<F>> {
    pub leaves: Vec<M>,
    pub digest_layers: Vec<DeviceBuffer<D>>,
    _marker: std::marker::PhantomData<F>,
}

impl<M: DeviceMatrix<BabyBear>, D: Copy> FieldMerkleTreeGpu<BabyBear, D, M> {
    pub fn new(hasher: &impl FieldMerkleTreeHasher<BabyBear, Digest = D>, leaves: Vec<M>) -> Self {
        let mut leaves_sorted: Vec<_> = leaves
            .iter()
            .map(|l| l.view())
            .sorted_by_key(|l| Reverse(l.height))
            .collect();

        let heights: Vec<usize> = leaves_sorted.iter().map(|m| m.height).collect(); 
        println!("heights = {:?}", heights);
        let max_height = heights[0];
        let log_max_height = log2_ceil_usize(max_height);

        let mut num_heights: Vec<usize> = vec![0; log_max_height + 2];
        let mut num_presums: Vec<usize> = Vec::with_capacity(log_max_height + 2);
        let mut height_offs: Vec<usize> = Vec::with_capacity(log_max_height + 2);
        let mut height_idx = 0;
        let mut matrix_count = 0;
        let mut sum_uniq_height = 0;
        for log_h in (0..=log_max_height+1) {
            num_presums.push(matrix_count);
            height_offs.push(sum_uniq_height);
            let height = 2usize.pow((log_max_height-log_h) as u32);
            while height_idx < heights.len() && heights[height_idx] == height {
                num_heights[log_h] += 1;
                height_idx += 1;
            }
            matrix_count += num_heights[log_h];
            sum_uniq_height += if num_heights[log_h] > 0 { height } else { 0 };
        }
        println!("num_heights = {:?}", num_heights);
        println!("num_presums = {:?}", num_presums);
        println!("height_offs = {:?}", height_offs);

        let num_heights_device = num_heights.to_device().unwrap();
        let num_presums_device = num_presums.to_device().unwrap();
        let height_offs_device = height_offs.to_device().unwrap();
        let leaves_sorted_device = leaves_sorted.to_device().unwrap();

        let mut all_digest_layers = DeviceBuffer::with_capacity(sum_uniq_height).unwrap();
        unsafe {
            all_digest_layers.set_len(sum_uniq_height);
            hasher.absorb_matrices(
                leaves_sorted_device.as_ptr(), 
                num_heights_device.as_ptr(), 
                num_presums_device.as_ptr(), 
                height_offs_device.as_ptr(), 
                log_max_height, 
                max_height, 
                all_digest_layers.as_mut_ptr(),
            );
        }

        // let all_digest_slice = all_digest_layers.as_slice_mut();
        // let first_digest_layer = &mut all_digest_slice[0..max_height];

        let mut leaves_largest_first = leaves
            .iter()
            .map(|l| l.view())
            .sorted_by_key(|l| Reverse(l.height))
            .peekable();

        let max_height = leaves_largest_first.peek().unwrap().height;
        let tallest_matrices = leaves_largest_first
            .peeking_take_while(|m| m.height == max_height)
            .collect_vec()
            .to_device()
            .unwrap();

        let mut first_digest_layer = DeviceBuffer::with_capacity(max_height).unwrap();
        unsafe {
            first_digest_layer.set_len(max_height);
            hasher.first_digest_layer(
                tallest_matrices.as_ptr(),
                tallest_matrices.len(),
                first_digest_layer.as_mut_ptr(),
                max_height,
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
                .collect_vec()
                .to_device()
                .unwrap();

            let mut next_digests = DeviceBuffer::<D>::with_capacity(next_layer_len).unwrap();
            unsafe {
                next_digests.set_len(next_layer_len);
                hasher.compress_and_inject(
                    prev_layer.as_ptr(),
                    matrices_to_inject.as_ptr(),
                    matrices_to_inject.len(),
                    next_digests.as_mut_ptr(),
                    next_layer_len,
                );
            }
            digest_layers.push(next_digests);
        }

        Self {
            leaves,
            digest_layers,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn root(&self) -> D {
        self.digest_layers.last().unwrap().to_host()[0]
    }
}

impl<F, W, M, const DIGEST_ELEMS: usize> ToHost for FieldMerkleTreeGpu<F, [W; DIGEST_ELEMS], M>
where
    F: Field,
    W: Copy,
    M: Send + Sync + DeviceMatrix<F> + ToHost<HostType = RowMajorMatrix<F>>,
{
    type HostType = FieldMerkleTree<F, W, RowMajorMatrix<F>, DIGEST_ELEMS>;

    fn to_host(&self) -> Self::HostType {
        let leaves = self.leaves.iter().map(|l| l.to_host()).collect::<Vec<_>>();
        let digest_layers = self
            .digest_layers
            .iter()
            .map(|l| l.to_host())
            .collect::<Vec<_>>();

        FieldMerkleTree::from_parts(leaves, digest_layers)
    }
}

impl<W, const DIGEST_ELEMS: usize> ToDevice
    for FieldMerkleTree<BabyBear, W, RowMajorMatrix<BabyBear>, DIGEST_ELEMS>
where
    BabyBear: Field,
    W: Copy,
{
    type DeviceType =
        FieldMerkleTreeGpu<BabyBear, [W; DIGEST_ELEMS], ColMajorMatrixDevice<BabyBear>>;

    fn to_device(&self) -> Result<Self::DeviceType, CudaError> {
        let leaves_device = self
            .leaves
            .iter()
            .map(|l| Ok(l.to_device()?.to_column_major()))
            .collect::<Result<Vec<_>, CudaError>>()?;

        let digest_layers_device = self
            .digest_layers
            .iter()
            .map(|l| l.to_device())
            .collect::<Result<Vec<_>, CudaError>>()?;

        Ok(FieldMerkleTreeGpu {
            leaves: leaves_device,
            digest_layers: digest_layers_device,
            _marker: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    pub mod baby_bear_tests {
        use crate::device::memory::{ToDevice, ToHost};
        use crate::matrix::{ColMajorMatrixDevice, RowMajorMatrixDevice};
        use crate::merkle_tree::{FieldMerkleTreeGpu, FieldMerkleTreeHasher};
        use crate::poseidon2::tests::baby_bear_tests::{
            poseidon2_baby_bear_16_compressor, poseidon2_baby_bear_16_hasher,
        };
        use crate::{
            device::DeviceBuffer,
            poseidon2::baby_bear::poseidon2_baby_bear_16_kernels::DIGEST_WIDTH,
            poseidon2::baby_bear::DeviceHasherBabyBear,
        };

        use p3_baby_bear::BabyBear;
        use p3_merkle_tree::FieldMerkleTree;

        pub type BabyBearFieldMerkleTreeGpu<M> =
            FieldMerkleTreeGpu<BabyBear, [BabyBear; DIGEST_WIDTH], M>;

        #[test]
        fn test_first_digest_layer() {
            let n = 1 << 16;
            let hasher = poseidon2_baby_bear_16_hasher();

            let (matrix_host_1, matrix_device_1) = RowMajorMatrixDevice::<BabyBear>::dummy(9, n);
            let (matrix_host_2, matrix_device_2) = RowMajorMatrixDevice::<BabyBear>::dummy(4, n);
            let tallest_matrices = vec![matrix_device_1.view(), matrix_device_2.view()]
                .to_device()
                .unwrap();
            let mut digests = DeviceBuffer::<[BabyBear; DIGEST_WIDTH]>::with_capacity(n).unwrap();
            let hasher_gpu = DeviceHasherBabyBear::new();
            unsafe {
                digests.set_len(n);
                hasher_gpu.first_digest_layer(
                    tallest_matrices.as_ptr(),
                    tallest_matrices.len(),
                    digests.as_mut_ptr(),
                    n,
                );
            }

            let tallest_matrices = vec![&matrix_host_1, &matrix_host_2];
            let digests_host = p3_merkle_tree::first_digest_layer(&hasher, tallest_matrices);

            let digests_device = digests.to_host();
            for i in 0..n {
                assert_eq!(digests_host[i], digests_device[i]);
            }
        }

        #[test]
        fn test_compress_and_inject() {
            let n = 1 << 16;
            let hasher = poseidon2_baby_bear_16_hasher();
            let compressor = poseidon2_baby_bear_16_compressor();

            let (matrix_host_1, matrix_device_1) = RowMajorMatrixDevice::<BabyBear>::dummy(9, n);
            let (matrix_host_2, matrix_device_2) =
                RowMajorMatrixDevice::<BabyBear>::dummy(4, n >> 1);

            let tallest_matrices = vec![matrix_device_1.view()].to_device().unwrap();
            let mut first_layer_digests =
                DeviceBuffer::<[BabyBear; DIGEST_WIDTH]>::with_capacity(n).unwrap();

            let hasher_gpu = DeviceHasherBabyBear::new();
            unsafe {
                first_layer_digests.set_len(n);
                hasher_gpu.first_digest_layer(
                    tallest_matrices.as_ptr(),
                    tallest_matrices.len(),
                    first_layer_digests.as_mut_ptr(),
                    n,
                );
            }

            let matrices_to_inject = vec![matrix_device_2.view()].to_device().unwrap();
            let mut next_digests =
                DeviceBuffer::<[BabyBear; DIGEST_WIDTH]>::with_capacity(n >> 1).unwrap();
            unsafe {
                next_digests.set_len(n / 2);
                hasher_gpu.compress_and_inject(
                    first_layer_digests.as_ptr(),
                    matrices_to_inject.as_ptr(),
                    matrices_to_inject.len(),
                    next_digests.as_mut_ptr(),
                    n,
                );
            }

            let tallest_matrices = vec![&matrix_host_1];
            let first_layer_digests_host =
                p3_merkle_tree::first_digest_layer(&hasher, tallest_matrices);

            let first_layer_digests_device = first_layer_digests.to_host();
            for i in 0..n {
                assert_eq!(first_layer_digests_host[i], first_layer_digests_device[i]);
            }

            let matrices_to_inject = vec![&matrix_host_2];
            let next_digests_host = p3_merkle_tree::compress_and_inject(
                &first_layer_digests_host,
                matrices_to_inject,
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
            let hasher = poseidon2_baby_bear_16_hasher();
            let compressor = poseidon2_baby_bear_16_compressor();

            let (matrix_host_1, matrix_device_1) = RowMajorMatrixDevice::<BabyBear>::dummy(600, n);

            let tallest_matrices = vec![matrix_device_1];
            let device_hasher = DeviceHasherBabyBear::default();
            let tree_device = BabyBearFieldMerkleTreeGpu::new(&device_hasher, tallest_matrices);
            let root_device = tree_device.root();

            let tallest_matrices = vec![matrix_host_1];
            let tree_host = FieldMerkleTree::new(&hasher, &compressor, tallest_matrices);
            let root_host: [BabyBear; DIGEST_WIDTH] = tree_host.root().into();

            assert_eq!(root_device, root_host);
        }

        #[test]
        fn test_col_major_commit_matrices() {
            let n = 1 << 16;
            let hasher = poseidon2_baby_bear_16_hasher();
            let compressor = poseidon2_baby_bear_16_compressor();

            let (matrix_host_1, matrix_device_1) = ColMajorMatrixDevice::<BabyBear>::dummy(600, n);

            let tallest_matrices = vec![matrix_device_1];
            let device_hasher = DeviceHasherBabyBear::default();
            let tree_device = BabyBearFieldMerkleTreeGpu::new(&device_hasher, tallest_matrices);
            let root_device = tree_device.root();

            let tallest_matrices = vec![matrix_host_1];
            let tree_host = FieldMerkleTree::new(&hasher, &compressor, tallest_matrices);
            let root_host: [BabyBear; DIGEST_WIDTH] = tree_host.root().into();

            assert_eq!(root_device, root_host);
        }
    }
    pub mod bn254_tests {
        use crate::device::memory::{ToDevice, ToHost};
        use crate::matrix::{ColMajorMatrixDevice, RowMajorMatrixDevice};
        use crate::merkle_tree::{FieldMerkleTreeGpu, FieldMerkleTreeHasher};
        use crate::poseidon2::tests::bn254_tests::{
            poseidon2_bn254_3_compressor, poseidon2_bn254_3_perm,
        };
        use crate::{
            device::DeviceBuffer, poseidon2::bn254::poseidon2_bn254_3_kernels::DIGEST_WIDTH,
            poseidon2::bn254::DeviceHasherBn254,
        };

        use p3_baby_bear::BabyBear;
        use p3_bn254_fr::{Bn254Fr, DiffusionMatrixBN254};
        use p3_merkle_tree::FieldMerkleTree;
        use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
        use p3_symmetric::MultiField32PaddingFreeSponge;

        pub type OuterVal = BabyBear;
        pub type OuterPerm =
            Poseidon2<Bn254Fr, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBN254, 3, 5>;
        pub type OuterHash = MultiField32PaddingFreeSponge<OuterVal, Bn254Fr, OuterPerm, 3, 16, 1>;

        pub type Bn254FieldMerkleTreeGpu<M> =
            FieldMerkleTreeGpu<BabyBear, [Bn254Fr; DIGEST_WIDTH], M>;

        #[test]
        fn test_first_digest_layer() {
            let n = 1 << 16;
            let perm = poseidon2_bn254_3_perm();
            let hasher = OuterHash::new(perm).unwrap();

            let (matrix_host_1, matrix_device_1) = RowMajorMatrixDevice::<BabyBear>::dummy(9, n);
            let (matrix_host_2, matrix_device_2) = RowMajorMatrixDevice::<BabyBear>::dummy(4, n);
            let tallest_matrices = vec![matrix_device_1.view(), matrix_device_2.view()]
                .to_device()
                .unwrap();
            let mut digests = DeviceBuffer::<[Bn254Fr; DIGEST_WIDTH]>::with_capacity(n).unwrap();
            let hasher_gpu = DeviceHasherBn254::new();
            unsafe {
                digests.set_len(n);
                hasher_gpu.first_digest_layer(
                    tallest_matrices.as_ptr(),
                    tallest_matrices.len(),
                    digests.as_mut_ptr(),
                    n,
                );
            }

            let tallest_matrices = vec![&matrix_host_1, &matrix_host_2];
            let digests_host = p3_merkle_tree::first_digest_layer(&hasher, tallest_matrices);

            let digests_device = digests.to_host();
            for i in 0..n {
                assert_eq!(digests_host[i], digests_device[i]);
            }
        }

        #[test]
        fn test_compress_and_inject() {
            let n = 1 << 16;
            let perm = poseidon2_bn254_3_perm();
            let hasher = OuterHash::new(perm).unwrap();
            let compressor = poseidon2_bn254_3_compressor();

            let (matrix_host_1, matrix_device_1) = RowMajorMatrixDevice::<BabyBear>::dummy(9, n);
            let (matrix_host_2, matrix_device_2) =
                RowMajorMatrixDevice::<BabyBear>::dummy(4, n >> 1);

            let tallest_matrices = vec![matrix_device_1.view()].to_device().unwrap();
            let mut first_layer_digests =
                DeviceBuffer::<[Bn254Fr; DIGEST_WIDTH]>::with_capacity(n).unwrap();

            let hasher_gpu = DeviceHasherBn254::new();
            unsafe {
                first_layer_digests.set_len(n);
                hasher_gpu.first_digest_layer(
                    tallest_matrices.as_ptr(),
                    tallest_matrices.len(),
                    first_layer_digests.as_mut_ptr(),
                    n,
                );
            }

            let matrices_to_inject = vec![matrix_device_2.view()].to_device().unwrap();
            let mut next_digests =
                DeviceBuffer::<[Bn254Fr; DIGEST_WIDTH]>::with_capacity(n >> 1).unwrap();
            unsafe {
                next_digests.set_len(n / 2);
                hasher_gpu.compress_and_inject(
                    first_layer_digests.as_ptr(),
                    matrices_to_inject.as_ptr(),
                    matrices_to_inject.len(),
                    next_digests.as_mut_ptr(),
                    n,
                );
            }

            let tallest_matrices = vec![&matrix_host_1];
            let first_layer_digests_host =
                p3_merkle_tree::first_digest_layer(&hasher, tallest_matrices);

            let first_layer_digests_device = first_layer_digests.to_host();
            for i in 0..n {
                assert_eq!(first_layer_digests_host[i], first_layer_digests_device[i]);
            }

            let matrices_to_inject = vec![&matrix_host_2];
            let next_digests_host = p3_merkle_tree::compress_and_inject(
                &first_layer_digests_host,
                matrices_to_inject,
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
            let perm = poseidon2_bn254_3_perm();
            let hasher = OuterHash::new(perm).unwrap();
            let compressor = poseidon2_bn254_3_compressor();

            let (matrix_host_1, matrix_device_1) = RowMajorMatrixDevice::<BabyBear>::dummy(600, n);

            let tallest_matrices = vec![matrix_device_1];
            let device_hasher = DeviceHasherBn254::default();
            let tree_device = Bn254FieldMerkleTreeGpu::new(&device_hasher, tallest_matrices);
            let root_device = tree_device.root();

            let tallest_matrices = vec![matrix_host_1];
            let tree_host = FieldMerkleTree::new(&hasher, &compressor, tallest_matrices);
            let root_host: [Bn254Fr; DIGEST_WIDTH] = tree_host.root().into();

            assert_eq!(root_device, root_host);
        }

        #[test]
        fn test_col_major_commit_matrices() {
            let n = 1 << 16;
            let perm = poseidon2_bn254_3_perm();
            let hasher = OuterHash::new(perm).unwrap();
            let compressor = poseidon2_bn254_3_compressor();

            let (matrix_host_1, matrix_device_1) = ColMajorMatrixDevice::<BabyBear>::dummy(600, n);

            let tallest_matrices = vec![matrix_device_1];
            let device_hasher = DeviceHasherBn254::default();
            let tree_device = Bn254FieldMerkleTreeGpu::new(&device_hasher, tallest_matrices);
            let root_device = tree_device.root();

            let tallest_matrices = vec![matrix_host_1];
            let tree_host = FieldMerkleTree::new(&hasher, &compressor, tallest_matrices);
            let root_host: [Bn254Fr; DIGEST_WIDTH] = tree_host.root().into();

            assert_eq!(root_device, root_host);
        }
    }
}
