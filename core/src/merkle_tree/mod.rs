use crate::{
    cuda_runtime::stream::CudaStream,
    device::{
        error::CudaError,
        memory::{ToDevice, ToHost},
        DeviceBuffer,
    },
    matrix::ColMajorMatrixDevice,
};

use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_field::Field;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_merkle_tree::MerkleTree;
use std::{cmp::Reverse, marker::PhantomData};

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
    pub fn new(
        hasher: &impl FieldMerkleTreeHasher<BabyBear, Digest = D>,
        leaves: Vec<M>,
        main_stream: &CudaStream,
    ) -> Self {
        let mut leaves_largest_first =
            leaves.iter().map(|l| l.view()).sorted_by_key(|l| Reverse(l.height)).peekable();

        let max_height = leaves_largest_first.peek().unwrap().height;
        let tallest_matrices = leaves_largest_first
            .peeking_take_while(|m| m.height == max_height)
            .collect_vec()
            .to_device_async(main_stream)
            .unwrap();

        let mut first_digest_layer =
            DeviceBuffer::with_capacity_in(max_height, main_stream).unwrap();
        unsafe {
            first_digest_layer.set_len(max_height);
            hasher.first_digest_layer(
                tallest_matrices.as_ptr(),
                tallest_matrices.len(),
                first_digest_layer.as_mut_ptr(),
                max_height,
                main_stream.handle(),
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
                .to_device_async(main_stream)
                .unwrap();

            let mut next_digests =
                DeviceBuffer::<D>::with_capacity_in(next_layer_len, main_stream).unwrap();
            unsafe {
                next_digests.set_len(next_layer_len);
                hasher.compress_and_inject(
                    prev_layer.as_ptr(),
                    matrices_to_inject.as_ptr(),
                    matrices_to_inject.len(),
                    next_digests.as_mut_ptr(),
                    next_layer_len,
                    main_stream.handle(),
                );
            }
            digest_layers.push(next_digests);
        }

        Self { leaves, digest_layers, _marker: std::marker::PhantomData }
    }

    pub fn root(&self) -> D {
        self.digest_layers.last().unwrap().to_host()[0]
    }
}

impl<D: Copy> FieldMerkleTreeGpu<BabyBear, D, ColMajorMatrixDevice<BabyBear>> {
    pub fn stream(&self) -> &CudaStream {
        self.leaves[0].stream()
    }
}

impl<F, W, M, const DIGEST_ELEMS: usize> ToHost for FieldMerkleTreeGpu<F, [W; DIGEST_ELEMS], M>
where
    F: Field,
    W: Copy,
    M: Send + Sync + DeviceMatrix<F> + ToHost<HostType = RowMajorMatrix<F>>,
{
    type HostType = MerkleTree<F, W, RowMajorMatrix<F>, DIGEST_ELEMS>;

    fn to_host(&self) -> Self::HostType {
        let leaves = self.leaves.iter().map(|l| l.to_host()).collect::<Vec<_>>();
        let digest_layers = self.digest_layers.iter().map(|l| l.to_host()).collect::<Vec<_>>();

        MerkleTree::<F, W, DenseMatrix<F>, DIGEST_ELEMS>::from_parts(leaves, digest_layers)
    }
}

impl<W, const DIGEST_ELEMS: usize> ToDevice
    for MerkleTree<BabyBear, W, RowMajorMatrix<BabyBear>, DIGEST_ELEMS>
where
    BabyBear: Field,
    W: Copy,
{
    type DeviceType =
        FieldMerkleTreeGpu<BabyBear, [W; DIGEST_ELEMS], ColMajorMatrixDevice<BabyBear>>;

    fn to_device_async(
        &self,
        stream: &crate::cuda_runtime::stream::CudaStream,
    ) -> Result<Self::DeviceType, CudaError> {
        let leaves_device = self
            .leaves
            .iter()
            .map(|l| Ok(l.to_device_async(stream)?.to_column_major()))
            .collect::<Result<Vec<_>, CudaError>>()?;

        let digest_layers_device = self
            .digest_layers
            .iter()
            .map(|l| l.to_device_async(stream))
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
        use crate::{
            cuda_runtime::stream::CudaStream,
            device::{
                memory::{ToDevice, ToHost},
                DeviceBuffer,
            },
            matrix::{ColMajorMatrixDevice, RowMajorMatrixDevice},
            merkle_tree::{FieldMerkleTreeGpu, FieldMerkleTreeHasher},
            poseidon2::{
                baby_bear::{poseidon2_baby_bear_16_kernels::DIGEST_WIDTH, DeviceHasherBabyBear},
                tests::baby_bear_tests::{
                    poseidon2_baby_bear_16_compressor, poseidon2_baby_bear_16_hasher,
                },
            },
        };

        use p3_baby_bear::BabyBear;
        use p3_field::Field;
        use p3_matrix::dense::RowMajorMatrix;
        use p3_merkle_tree::MerkleTree;
        use p3_poseidon2::Poseidon2;
        use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

        pub type BabyBearFieldMerkleTreeGpu<M> =
            FieldMerkleTreeGpu<BabyBear, [BabyBear; DIGEST_WIDTH], M>;

        #[test]
        fn test_first_digest_layer() {
            let n = 1 << 16;
            let hasher = poseidon2_baby_bear_16_hasher();

            let (matrix_host_1, matrix_device_1) = RowMajorMatrixDevice::<BabyBear>::dummy(9, n);
            let (matrix_host_2, matrix_device_2) = RowMajorMatrixDevice::<BabyBear>::dummy(4, n);
            let tallest_matrices =
                vec![matrix_device_1.view(), matrix_device_2.view()].to_device().unwrap();
            let mut digests = DeviceBuffer::<[BabyBear; DIGEST_WIDTH]>::with_capacity(n).unwrap();
            let hasher_gpu = DeviceHasherBabyBear::new();
            unsafe {
                digests.set_len(n);
                hasher_gpu.first_digest_layer(
                    tallest_matrices.as_ptr(),
                    tallest_matrices.len(),
                    digests.as_mut_ptr(),
                    n,
                    CudaStream::default().handle(),
                );
            }

            let tallest_matrices = vec![&matrix_host_1, &matrix_host_2];
            let digests_host = p3_merkle_tree::first_digest_layer::<
                <BabyBear as Field>::Packing,                           // P
                <BabyBear as Field>::Packing,                           // PW
                PaddingFreeSponge<Poseidon2<_, _, _, 16, 7>, 16, 8, 8>, // H
                RowMajorMatrix<BabyBear>,                               // M
                8,                                                      // DIGEST_ELEMS
            >(&hasher, tallest_matrices);

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
                    CudaStream::default().handle(),
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
                    CudaStream::default().handle(),
                );
            }

            let tallest_matrices = vec![&matrix_host_1];
            let first_layer_digests_host = p3_merkle_tree::first_digest_layer::<
                <BabyBear as Field>::Packing,
                <BabyBear as Field>::Packing,
                PaddingFreeSponge<Poseidon2<_, _, _, 16, 7>, 16, 8, 8>,
                RowMajorMatrix<BabyBear>,
                8,
            >(&hasher, tallest_matrices);

            let first_layer_digests_device = first_layer_digests.to_host();
            for i in 0..n {
                assert_eq!(first_layer_digests_host[i], first_layer_digests_device[i]);
            }

            let matrices_to_inject = vec![&matrix_host_2];
            let next_digests_host =
                p3_merkle_tree::compress_and_inject::<
                    <BabyBear as Field>::Packing,
                    <BabyBear as Field>::Packing,
                    PaddingFreeSponge<Poseidon2<_, _, _, 16, 7>, 16, 8, 8>,
                    TruncatedPermutation<Poseidon2<_, _, _, 16, 7>, 2, 8, 16>,
                    RowMajorMatrix<BabyBear>,
                    8,
                >(
                    &first_layer_digests_host, matrices_to_inject, &hasher, &compressor
                );

            // &hasher, &compressor, &first_layer_digests_host, matrices_to_inject

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
            let tree_device = BabyBearFieldMerkleTreeGpu::new(
                &device_hasher,
                tallest_matrices,
                &CudaStream::default(),
            );
            let root_device = tree_device.root();

            let tallest_matrices = vec![matrix_host_1];
            let tree_host = MerkleTree::new::<
                <BabyBear as Field>::Packing,
                <BabyBear as Field>::Packing,
                PaddingFreeSponge<Poseidon2<_, _, _, 16, 7>, 16, 8, 8>,
                TruncatedPermutation<Poseidon2<_, _, _, 16, 7>, 2, 8, 16>,
            >(&hasher, &compressor, tallest_matrices);
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
            let tree_device = BabyBearFieldMerkleTreeGpu::new(
                &device_hasher,
                tallest_matrices,
                &CudaStream::default(),
            );
            let root_device = tree_device.root();

            let tallest_matrices = vec![matrix_host_1];
            let tree_host = MerkleTree::new::<
                <BabyBear as Field>::Packing,
                <BabyBear as Field>::Packing,
                PaddingFreeSponge<Poseidon2<_, _, _, 16, 7>, 16, 8, 8>,
                TruncatedPermutation<Poseidon2<_, _, _, 16, 7>, 2, 8, 16>,
            >(&hasher, &compressor, tallest_matrices);
            let root_host: [BabyBear; DIGEST_WIDTH] = tree_host.root().into();

            assert_eq!(root_device, root_host);
        }
    }
    pub mod bn254_tests {
        use crate::{
            cuda_runtime::stream::CudaStream,
            device::{
                memory::{ToDevice, ToHost},
                DeviceBuffer,
            },
            matrix::{ColMajorMatrixDevice, RowMajorMatrixDevice},
            merkle_tree::{FieldMerkleTreeGpu, FieldMerkleTreeHasher},
            poseidon2::{
                bn254::{poseidon2_bn254_3_kernels::DIGEST_WIDTH, DeviceHasherBn254},
                tests::bn254_tests::{poseidon2_bn254_3_compressor, poseidon2_bn254_3_perm},
            },
        };

        use p3_baby_bear::BabyBear;
        use p3_bn254_fr::{Bn254Fr, Poseidon2Bn254};
        use p3_merkle_tree::MerkleTree;
        use p3_poseidon2::{ExternalLayerConstants, Poseidon2};
        use p3_symmetric::MultiField32PaddingFreeSponge;

        pub type OuterVal = BabyBear;
        pub type OuterPerm = Poseidon2Bn254<3>;
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
            let tallest_matrices =
                vec![matrix_device_1.view(), matrix_device_2.view()].to_device().unwrap();
            let mut digests = DeviceBuffer::<[Bn254Fr; DIGEST_WIDTH]>::with_capacity(n).unwrap();
            let hasher_gpu = DeviceHasherBn254::new();
            unsafe {
                digests.set_len(n);
                hasher_gpu.first_digest_layer(
                    tallest_matrices.as_ptr(),
                    tallest_matrices.len(),
                    digests.as_mut_ptr(),
                    n,
                    CudaStream::default().handle(),
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
                    CudaStream::default().handle(),
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
                    CudaStream::default().handle(),
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
            let tree_device = Bn254FieldMerkleTreeGpu::new(
                &device_hasher,
                tallest_matrices,
                &CudaStream::default(),
            );
            let root_device = tree_device.root();

            let tallest_matrices = vec![matrix_host_1];
            let tree_host = MerkleTree::new(&hasher, &compressor, tallest_matrices);
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
            let tree_device = Bn254FieldMerkleTreeGpu::new(
                &device_hasher,
                tallest_matrices,
                &CudaStream::default(),
            );
            let root_device = tree_device.root();

            let tallest_matrices = vec![matrix_host_1];
            let tree_host = MerkleTree::new(&hasher, &compressor, tallest_matrices);
            let root_host: [Bn254Fr; DIGEST_WIDTH] = tree_host.root().into();

            assert_eq!(root_device, root_host);
        }
    }
}
