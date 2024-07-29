pub(super) mod ffi;
mod permutation;
mod prover;
mod quotient;
mod utils;

use p3_challenger::{CanObserve, CanSample, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::Radix2DitParallel;
use p3_fri::{FriConfig, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
pub use permutation::*;
pub use prover::*;
pub use quotient::*;
pub use utils::*;

use p3_baby_bear::BabyBear;
use sp1_core::{
    stark::{Com, PcsProverData, StarkGenericConfig},
    utils::BabyBearPoseidon2,
};

type EF = <BabyBearPoseidon2 as StarkGenericConfig>::Challenge;

pub type PcsConfig<SC: BabyBearFriConfig> =
    FriConfig<ExtensionMmcs<SC::Val, SC::Challenge, SC::ValMmcs>>;

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
        + GrindingChallenger<Witness = BabyBear>
        + FieldChallenger<BabyBear>;
}

pub trait BabyBearPoseidon2Config:
    StarkGenericConfig<
    Val = BabyBear,
    Challenge = <BabyBearPoseidon2 as StarkGenericConfig>::Challenge,
    Challenger = <BabyBearPoseidon2 as StarkGenericConfig>::Challenger,
    Pcs = <BabyBearPoseidon2 as StarkGenericConfig>::Pcs,
>
{
}

impl<SC> BabyBearPoseidon2Config for SC where
    SC: StarkGenericConfig<
        Val = BabyBear,
        Challenge = <BabyBearPoseidon2 as StarkGenericConfig>::Challenge,
        Challenger = <BabyBearPoseidon2 as StarkGenericConfig>::Challenger,
        Pcs = <BabyBearPoseidon2 as StarkGenericConfig>::Pcs,
    >
{
}

impl<SC> BabyBearFriConfig for SC
where
    SC: BabyBearPoseidon2Config,
{
    type ValMmcs = sp1_core::utils::baby_bear_poseidon2::ValMmcs;
    type RowMajorProverData = PcsProverData<SC>;
    type FriChallenger = SC::Challenger;
}
