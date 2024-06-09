mod permutation;
mod quotient;

pub use permutation::*;

pub(super) mod ffi {
    use super::{
        quotient::{LagrangeSelectorsView, TwoAdicMultiplicativeCosetDevice},
        DeviceInteractionsView,
    };
    use crate::matrix::{MatrixViewDevice, MatrixViewMutDevice};
    use air::operation::Operation;
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    extern "C" {
        pub fn populate_permutation_rows(
            interactions: DeviceInteractionsView<BabyBear>,
            permutation: MatrixViewMutDevice<BinomialExtensionField<BabyBear, 4>>,
            preprocessed: MatrixViewDevice<BabyBear>,
            main: MatrixViewDevice<BabyBear>,
            alpha: BinomialExtensionField<BabyBear, 4>,
            beta: BinomialExtensionField<BabyBear, 4>,
            batch_size: usize,
            num_blocks: usize,
            num_threads_per_block: usize,
        );

        pub fn quotient_values(
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
        );
    }
}
