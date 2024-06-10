pub(super) mod ffi;
mod permutation;
mod prover;
mod quotient;
mod quotient_host;
mod trace;
mod utils;

pub use permutation::*;
pub use prover::*;
pub use quotient_host::*;
use sp1_core::{stark::StarkGenericConfig, utils::BabyBearPoseidon2};
pub use trace::*;
pub use utils::*;

use p3_baby_bear::BabyBear;

pub(super) mod quotient_gpu {
    use super::quotient::{LagrangeSelectorsView, TwoAdicMultiplicativeCosetDevice};
    use crate::matrix::MatrixViewDevice;
    use air::operation::Operation;
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    #[link_name = "quotient_gpu"]
    #[allow(unused_attributes)]
    extern "C" {
        #[link_name = "computeValues"]
        #[allow(unused)]
        pub fn compute_values(
            chip_id: usize,
            eval_program: *const Operation,
            eval_program_len: usize,
            cumulative_sum: BinomialExtensionField<BabyBear, 4>,
            trace_domain: TwoAdicMultiplicativeCosetDevice<BabyBear>,
            quotient_domain: TwoAdicMultiplicativeCosetDevice<BabyBear>,
            preprocessed_trace_on_quotient_domain: MatrixViewDevice<BabyBear>,
            main_trace_on_quotient_domain: MatrixViewDevice<BabyBear>,
            permutation_trace_on_quotient_domain: MatrixViewDevice<BabyBear>,
            perm_challenges: *const BinomialExtensionField<BabyBear, 4>,
            alpha: BinomialExtensionField<BabyBear, 4>,
            public_values: *const BabyBear,
            selectors: LagrangeSelectorsView<BabyBear>,
            quotient_values: *mut BinomialExtensionField<BabyBear, 4>,
            num_blocks: usize,
            num_threads_per_block: usize,
        );
    }
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
