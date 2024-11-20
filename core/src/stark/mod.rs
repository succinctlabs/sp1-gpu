mod error;
pub(super) mod ffi;
mod permutation;
mod prover;
mod quotient;
mod utils;

pub use error::*;
use p3_bn254_fr::Bn254Fr;
pub use permutation::*;
pub use prover::*;
pub use quotient::*;
use sp1_stark::{
    baby_bear_poseidon2::{BabyBearPoseidon2, DigestHash, Val, ValMmcs},
    Com, PcsProverData, StarkGenericConfig, DIGEST_SIZE,
};
pub use utils::*;

use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::Radix2DitParallel;
use p3_field::AbstractField;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use sp1_recursion_core::stark::{
    BabyBearPoseidon2Outer, OuterDigestHash, OuterValMmcs, DIGEST_SIZE as OUTER_DIGEST_SIZE,
};

use p3_baby_bear::BabyBear;

use crate::challenger::grind_on_device::GrindOnDevice;

type EF = <BabyBearPoseidon2 as StarkGenericConfig>::Challenge;

pub type PcsConfig<SC> = FriConfig<
    ExtensionMmcs<
        <SC as StarkGenericConfig>::Val,
        <SC as StarkGenericConfig>::Challenge,
        <SC as BabyBearFriConfig>::ValMmcs,
    >,
>;

pub type FriMmcs<SC> = ExtensionMmcs<BabyBear, EF, <SC as BabyBearFriConfig>::ValMmcs>;

pub trait BabyBearFriConfig:
    StarkGenericConfig<
    Val = BabyBear,
    Challenge = EF,
    Challenger = Self::FriChallenger,
    Pcs = TwoAdicFriPcs<
        BabyBear,
        Radix2DitParallel,
        Self::ValMmcs,
        ExtensionMmcs<BabyBear, EF, Self::ValMmcs>,
    >,
>
{
    type ValMmcs: Mmcs<BabyBear, ProverData<RowMajorMatrix<BabyBear>> = Self::RowMajorProverData>;
    type RowMajorProverData: Clone;
    type FriChallenger: CanObserve<<Self::ValMmcs as Mmcs<BabyBear>>::Commitment>
        + CanSample<EF>
        + GrindOnDevice<Witness = BabyBear>
        + FieldChallenger<BabyBear>;

    fn zero_commitment() -> Com<Self>;
}

impl BabyBearFriConfig for BabyBearPoseidon2 {
    type ValMmcs = ValMmcs;
    type RowMajorProverData = PcsProverData<Self>;
    type FriChallenger = <Self as StarkGenericConfig>::Challenger;

    fn zero_commitment() -> Com<BabyBearPoseidon2> {
        DigestHash::from([Val::zero(); DIGEST_SIZE])
    }
}

impl BabyBearFriConfig for BabyBearPoseidon2Outer {
    type ValMmcs = OuterValMmcs;
    type RowMajorProverData = PcsProverData<Self>;
    type FriChallenger = <Self as StarkGenericConfig>::Challenger;

    fn zero_commitment() -> Com<BabyBearPoseidon2Outer> {
        OuterDigestHash::from([Bn254Fr::zero(); OUTER_DIGEST_SIZE])
    }
}
