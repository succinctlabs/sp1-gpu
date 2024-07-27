use p3_baby_bear::BabyBear;
use p3_commit::Mmcs;
use p3_field::{PackedField, PackedValue};
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{de::DeserializeOwned, Serialize};

use crate::matrix::ColMajorMatrixDevice;

use super::{FieldMerkleTreeGpu, FieldMerkleTreeHasher};

pub trait MmcsCommitter<T: Send + Sync, M: Mmcs<T>> {
    type ProverData;
    type Matrix;

    fn commit(&self, matrices: Vec<Self::Matrix>) -> (M::Commitment, Self::ProverData);
}

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

    fn commit(
        &self,
        matrices: Vec<Self::Matrix>,
    ) -> (Hash<P::Scalar, PW::Value, DIGEST_ELEMS>, Self::ProverData) {
        let merkle_tree = FieldMerkleTreeGpu::new(&self.hasher, matrices);
        let root = merkle_tree.root().into();

        (root, merkle_tree)
    }
}
