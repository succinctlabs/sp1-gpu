use p3_baby_bear::BabyBear;
use p3_commit::Mmcs;
use p3_field::{PackedField, PackedValue};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    matrix::{ColMajorMatrixDevice, DeviceMatrix},
    poseidon2::{baby_bear::DeviceHasherBabyBear, bn254::DeviceHasherBn254},
};

use super::{FieldMerkleTreeGpu, FieldMerkleTreeHasher};

pub type CommitterProverData<T, M, C> = <C as MmcsCommitter<T, M>>::ProverData;

pub trait MmcsCommitter<T: Send + Sync, M: Mmcs<T>> {
    type ProverData: MmcsProverData<Self::Matrix>;
    type Matrix;

    fn commit(&self, matrices: Vec<Self::Matrix>) -> (M::Commitment, Self::ProverData);
}

pub trait MmcsProverData<Matrix> {
    fn matrices(&self) -> &[Matrix];

    fn matrices_mut(&mut self) -> &mut [Matrix];

    fn clear_matrices(&mut self);

    fn push_matrix(&mut self, matrix: Matrix);
}

pub type Poseidon2BabyBearCommitter = FieldMerkleTreeDeviceCommitter<DeviceHasherBabyBear>;
pub type Poseidon2Bn254Committer = FieldMerkleTreeDeviceCommitter<DeviceHasherBn254>;

#[derive(Debug, Clone, Copy, Default)]
pub struct FieldMerkleTreeDeviceCommitter<H> {
    hasher: H,
}

impl<Hasher, P, PW, H, C, const DIGEST_ELEMS: usize>
    MmcsCommitter<BabyBear, FieldMerkleTreeMmcs<P, PW, H, C, DIGEST_ELEMS>>
    for FieldMerkleTreeDeviceCommitter<Hasher>
where
    Hasher: FieldMerkleTreeHasher<BabyBear, Digest = [PW::Value; DIGEST_ELEMS]>,
    P: PackedField<Scalar = BabyBear>,
    PW: PackedValue,
    H: CryptographicHasher<P::Scalar, [PW::Value; DIGEST_ELEMS]>,
    H: CryptographicHasher<P, [PW; DIGEST_ELEMS]>,
    H: Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], 2>,
    C: PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>,
    C: Sync,
    PW::Value: Eq,
    [PW::Value; DIGEST_ELEMS]: Serialize + DeserializeOwned,
{
    type Matrix = ColMajorMatrixDevice<BabyBear>;
    type ProverData = FieldMerkleTreeGpu<BabyBear, [PW::Value; DIGEST_ELEMS], Self::Matrix>;

    #[inline]
    fn commit(
        &self,
        matrices: Vec<Self::Matrix>,
    ) -> (Hash<P::Scalar, PW::Value, DIGEST_ELEMS>, Self::ProverData) {
        let merkle_tree = FieldMerkleTreeGpu::new(&self.hasher, matrices);
        let root = merkle_tree.root().into();

        (root, merkle_tree)
    }
}

impl<F: Copy, D: Copy, M: DeviceMatrix<F>> MmcsProverData<M> for FieldMerkleTreeGpu<F, D, M> {
    #[inline]
    fn matrices(&self) -> &[M] {
        &self.leaves
    }

    #[inline]
    fn matrices_mut(&mut self) -> &mut [M] {
        &mut self.leaves
    }

    #[inline]
    fn clear_matrices(&mut self) {
        self.leaves.clear();
    }

    #[inline]
    fn push_matrix(&mut self, matrix: M) {
        self.leaves.push(matrix);
    }
}
