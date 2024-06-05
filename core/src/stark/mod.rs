mod permutation;
mod quotient;

pub use permutation::*;

pub(super) mod ffi {
    use super::{quotient::LagrangeSelectorsView, DeviceInteractionsView};
    use crate::matrix::{MatrixViewDevice, MatrixViewMutDevice};
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
            cumulative_sum: BinomialExtensionField<BabyBear, 4>,
            trace_domain: MatrixViewDevice<BabyBear>,
            quotient_domain: MatrixViewDevice<BabyBear>,
            preprocessed_trace_on_quotient_domain: MatrixViewDevice<BabyBear>,
            main_trace_on_quotient_domain: MatrixViewDevice<BabyBear>,
            permutation_trace_on_quotient_domain: MatrixViewDevice<BabyBear>,
            perm_challenges: *const BinomialExtensionField<BabyBear, 4>,
            alpha: BinomialExtensionField<BabyBear, 4>,
            public_values: BinomialExtensionField<BabyBear, 4>,
            quotient_values: BinomialExtensionField<BabyBear, 4>,
            selectors: LagrangeSelectorsView<BabyBear>,
            n_blocks: usize,
            n_threads_per_block: usize,
        );
    }
}
