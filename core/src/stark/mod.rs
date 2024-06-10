pub(super) mod ffi;
mod permutation;
mod prover;
mod quotient;
mod trace;
mod utils;

pub use permutation::*;
pub use prover::*;
pub use quotient::*;
pub use trace::*;
pub use utils::*;

use p3_baby_bear::BabyBear;
use sp1_core::{stark::StarkGenericConfig, utils::BabyBearPoseidon2};

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
