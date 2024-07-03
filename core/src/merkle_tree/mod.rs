use crate::device::buffer::DeviceBuffer;
use crate::device::buffer::SyncBuffer;
use crate::device::memory::ToDevice;
use crate::device::memory::ToHost;
use crate::device::CudaSync;
use crate::matrix::ColMajorMatrixDevice;
use crate::matrix::RowMajorMatrixDevice;
use crate::poseidon2::baby_bear_gpu::poseidon2_baby_bear_16_kernels::DIGEST_WIDTH;
use crate::poseidon2::baby_bear_gpu::HasherBabyBearGPU;

use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::FieldMerkleTree;
use std::cmp::Reverse;
use std::marker::PhantomData;

use crate::matrix::DeviceMatrix;

pub struct FieldMerkleTreeGpu<F: Copy, D: Copy, M: DeviceMatrix<F> = RowMajorMatrixDevice<F>> {
    pub leaves: Vec<M>,
    pub digest_layers: Vec<SyncBuffer<D>>,
    _marker: std::marker::PhantomData<F>,
}

impl<M: DeviceMatrix<BabyBear>> FieldMerkleTreeGpu<BabyBear, [BabyBear; DIGEST_WIDTH], M> {
    pub fn new(leaves: Vec<M>) -> Self {
        let mut leaves_largest_first = leaves
            .iter()
            .map(|l| l.view())
            .sorted_by_key(|l| Reverse(l.height))
            .peekable();

        let max_height = leaves_largest_first.peek().unwrap().height;
        let tallest_matrices = leaves_largest_first
            .peeking_take_while(|m| m.height == max_height)
            .collect_vec()
            .to_device();

        let mut first_digest_layer = DeviceBuffer::with_capacity(max_height);
        let hasher = HasherBabyBearGPU::new();
        unsafe {
            first_digest_layer.set_len(max_height);
            hasher.first_digest_layer(
                tallest_matrices.as_ptr(),
                tallest_matrices.len(),
                first_digest_layer.as_mut_ptr(),
                max_height / 32 + 1,
                32,
            );
        }

        let first_digest_layer = CudaSync::new(first_digest_layer).unwrap();
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
                .to_device();

            let mut next_digests =
                DeviceBuffer::<[BabyBear; DIGEST_WIDTH]>::with_capacity(next_layer_len);
            unsafe {
                next_digests.set_len(next_layer_len);
                hasher.compress_and_inject(
                    prev_layer.as_ptr(),
                    prev_layer.len(),
                    matrices_to_inject.as_ptr(),
                    matrices_to_inject.len(),
                    next_digests.as_mut_ptr(),
                    next_layer_len / 32 + 1,
                    32,
                );
            }
            digest_layers.push(CudaSync::new(next_digests).unwrap());
        }

        Self {
            leaves,
            digest_layers,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn root(&self) -> [BabyBear; DIGEST_WIDTH] {
        self.digest_layers.last().unwrap().to_host()[0]
    }
}

impl<F, M> ToHost for FieldMerkleTreeGpu<F, [F; DIGEST_WIDTH], M>
where
    F: Field,
    M: Send + Sync + DeviceMatrix<F> + ToHost<HostType = RowMajorMatrix<F>>,
{
    type HostType = FieldMerkleTree<F, F, RowMajorMatrix<F>, DIGEST_WIDTH>;

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

impl ToDevice for FieldMerkleTree<BabyBear, BabyBear, RowMajorMatrix<BabyBear>, DIGEST_WIDTH> {
    type DeviceType = FieldMerkleTreeGpu<
        BabyBear,
        [BabyBear; DIGEST_WIDTH],
        CudaSync<ColMajorMatrixDevice<BabyBear>>,
    >;

    fn to_device(&self) -> Self::DeviceType {
        let leaves_device = self
            .leaves
            .iter()
            .map(|l| CudaSync::new(l.to_device().to_column_major()).unwrap())
            .collect::<Vec<_>>();

        let digest_layers_device = self
            .digest_layers
            .iter()
            .map(|l| l.to_device_sync().unwrap())
            .collect::<Vec<_>>();

        FieldMerkleTreeGpu {
            leaves: leaves_device,
            digest_layers: digest_layers_device,
            _marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    pub mod baby_bear_tests {
        use crate::device::memory::{ToDevice, ToHost};
        use crate::matrix::{ColMajorMatrixDevice, RowMajorMatrixDevice};
        use crate::merkle_tree::FieldMerkleTreeGpu;
        use crate::poseidon2::tests::baby_bear_tests::{
            poseidon2_baby_bear_16_compressor, poseidon2_baby_bear_16_hasher,
        };
        use crate::{
            device::buffer::DeviceBuffer,
            poseidon2::baby_bear_gpu::poseidon2_baby_bear_16_kernels::DIGEST_WIDTH,
            poseidon2::baby_bear_gpu::HasherBabyBearGPU,
        };

        use p3_baby_bear::BabyBear;
        use p3_merkle_tree::FieldMerkleTree;

        #[test]
        fn test_first_digest_layer() {
            let n = 1 << 16;
            let hasher = poseidon2_baby_bear_16_hasher();

            let (matrix_host_1, matrix_device_1) = RowMajorMatrixDevice::<BabyBear>::dummy(9, n);
            let (matrix_host_2, matrix_device_2) = RowMajorMatrixDevice::<BabyBear>::dummy(4, n);
            let tallest_matrices = vec![matrix_device_1.view(), matrix_device_2.view()].to_device();
            let mut digests = DeviceBuffer::<[BabyBear; DIGEST_WIDTH]>::with_capacity(n);
            let hasher_gpu = HasherBabyBearGPU::new();
            unsafe {
                digests.set_len(n);
                hasher_gpu.first_digest_layer(
                    tallest_matrices.as_ptr(),
                    tallest_matrices.len(),
                    digests.as_mut_ptr(),
                    n / 32,
                    32,
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

            let tallest_matrices = vec![matrix_device_1.view()].to_device();
            let mut first_layer_digests =
                DeviceBuffer::<[BabyBear; DIGEST_WIDTH]>::with_capacity(n);

            let hasher_gpu = HasherBabyBearGPU::new();
            unsafe {
                first_layer_digests.set_len(n);
                hasher_gpu.first_digest_layer(
                    tallest_matrices.as_ptr(),
                    tallest_matrices.len(),
                    first_layer_digests.as_mut_ptr(),
                    n / 32,
                    32,
                );
            }

            let matrices_to_inject = vec![matrix_device_2.view()].to_device();
            let mut next_digests = DeviceBuffer::<[BabyBear; DIGEST_WIDTH]>::with_capacity(n >> 1);
            unsafe {
                next_digests.set_len(n / 2);
                hasher_gpu.compress_and_inject(
                    first_layer_digests.as_ptr(),
                    first_layer_digests.len(),
                    matrices_to_inject.as_ptr(),
                    matrices_to_inject.len(),
                    next_digests.as_mut_ptr(),
                    n / 32,
                    32,
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

            let start = std::time::Instant::now();
            let tallest_matrices = vec![matrix_device_1];
            let tree_device = FieldMerkleTreeGpu::new(tallest_matrices);
            let root_device = tree_device.root();
            println!("time: {:?}", start.elapsed().as_secs_f64());

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

            let start = std::time::Instant::now();
            let tallest_matrices = vec![matrix_device_1];
            let tree_device = FieldMerkleTreeGpu::new(tallest_matrices);
            let root_device = tree_device.root();
            println!("Device time: {:?}", start.elapsed());

            let tallest_matrices = vec![matrix_host_1];
            let tree_host = FieldMerkleTree::new(&hasher, &compressor, tallest_matrices);
            let root_host: [BabyBear; DIGEST_WIDTH] = tree_host.root().into();

            assert_eq!(root_device, root_host);
        }
    }
    pub mod bn254_tests {
        use crate::device::memory::{ToDevice, ToHost};
        use crate::matrix::{ColMajorMatrixDevice, RowMajorMatrixDevice};
        use crate::merkle_tree::FieldMerkleTreeGpu;
        use crate::poseidon2::tests::bn254_tests::{
            poseidon2_bn254_3_compressor, poseidon2_bn254_3_perm,
        };
        use crate::{
            device::buffer::DeviceBuffer,
            poseidon2::bn254_gpu::poseidon2_bn254_3_kernels::DIGEST_WIDTH,
            poseidon2::bn254_gpu::HasherBn254GPU,
        };
        use p3_baby_bear::BabyBear;
        use p3_bn254_fr::{Bn254Fr, DiffusionMatrixBN254};
        // use p3_merkle_tree::FieldMerkleTree;
        use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
        use p3_symmetric::MultiField32PaddingFreeSponge;

        pub type OuterVal = BabyBear;
        pub type OuterPerm =
            Poseidon2<Bn254Fr, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBN254, 3, 5>;
        pub type OuterHash = MultiField32PaddingFreeSponge<OuterVal, Bn254Fr, OuterPerm, 3, 16, 1>;

        #[test]
        fn test_first_digest_layer() {
            let n = 1 << 16;
            let perm = poseidon2_bn254_3_perm();
            let hasher = OuterHash::new(perm).unwrap();

            // GPU implementation gets at least two things wrong:
            // 1. Differs from CPU when the first tallest matrix has width not multiple of 8 (if there is a second tallest matrix)
            // 2. Differs in reduce32 implementation

            let (matrix_host_1, matrix_device_1) = RowMajorMatrixDevice::<BabyBear>::dummy(9, n);
            let (matrix_host_2, matrix_device_2) = RowMajorMatrixDevice::<BabyBear>::dummy(4, n);
            let tallest_matrices = vec![matrix_device_1.view(), matrix_device_2.view()].to_device();
            // let tallest_matrices = v\ec![matrix_device_1.view()].to_device();
            let mut digests = DeviceBuffer::<[Bn254Fr; DIGEST_WIDTH]>::with_capacity(n);
            let hasher_gpu = HasherBn254GPU::new();
            unsafe {
                digests.set_len(n);
                hasher_gpu.first_digest_layer(
                    tallest_matrices.as_ptr(),
                    tallest_matrices.len(),
                    digests.as_mut_ptr(),
                    n / 32,
                    32,
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

            let tallest_matrices = vec![matrix_device_1.view()].to_device();
            let mut first_layer_digests = DeviceBuffer::<[Bn254Fr; DIGEST_WIDTH]>::with_capacity(n);

            let hasher_gpu = HasherBn254GPU::new();
            unsafe {
                first_layer_digests.set_len(n);
                hasher_gpu.first_digest_layer(
                    tallest_matrices.as_ptr(),
                    tallest_matrices.len(),
                    first_layer_digests.as_mut_ptr(),
                    n / 32,
                    32,
                );
            }

            let matrices_to_inject = vec![matrix_device_2.view()].to_device();
            let mut next_digests = DeviceBuffer::<[Bn254Fr; DIGEST_WIDTH]>::with_capacity(n >> 1);
            unsafe {
                next_digests.set_len(n / 2);
                hasher_gpu.compress_and_inject(
                    first_layer_digests.as_ptr(),
                    first_layer_digests.len(),
                    matrices_to_inject.as_ptr(),
                    matrices_to_inject.len(),
                    next_digests.as_mut_ptr(),
                    n / 32,
                    32,
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

        // #[test]
        // fn test_commit_matrices() {
        //     let n = 1 << 16;
        //     let hasher = poseidon2_bn254_3_hasher();
        //     let compressor = poseidon2_bn254_3_compressor();

        //     let (matrix_host_1, matrix_device_1) = RowMajorMatrixDevice::<Bn254Fr>::dummy(600, n);

        //     let start = std::time::Instant::now();
        //     let tallest_matrices = vec![matrix_device_1];
        //     let tree_device = FieldMerkleTreeGpu::new(tallest_matrices);
        //     let root_device = tree_device.root();
        //     println!("time: {:?}", start.elapsed().as_secs_f64());

        //     let tallest_matrices = vec![matrix_host_1];
        //     let tree_host = FieldMerkleTree::new(&hasher, &compressor, tallest_matrices);
        //     let root_host: [Bn254Fr; DIGEST_WIDTH] = tree_host.root().into();

        //     assert_eq!(root_device, root_host);
        // }

        // #[test]
        // fn test_col_major_commit_matrices() {
        //     let n = 1 << 16;
        //     let hasher = poseidon2_bn254_3_hasher();
        //     let compressor = poseidon2_bn254_3_compressor();

        //     let (matrix_host_1, matrix_device_1) = ColMajorMatrixDevice::<Bn254Fr>::dummy(600, n);

        //     let start = std::time::Instant::now();
        //     let tallest_matrices = vec![matrix_device_1];
        //     let tree_device = FieldMerkleTreeGpu::new(tallest_matrices);
        //     let root_device = tree_device.root();
        //     println!("Device time: {:?}", start.elapsed());

        //     let tallest_matrices = vec![matrix_host_1];
        //     let tree_host = FieldMerkleTree::new(&hasher, &compressor, tallest_matrices);
        //     let root_host: [Bn254Fr; DIGEST_WIDTH] = tree_host.root().into();

        //     assert_eq!(root_device, root_host);
        // }
    }
}
