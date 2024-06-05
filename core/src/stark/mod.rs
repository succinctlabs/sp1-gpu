mod permutation;
mod prover;

pub use permutation::*;
pub use prover::*;

pub(super) mod ffi {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use crate::matrix::{MatrixViewDevice, MatrixViewMutDevice};

    use super::DeviceInteractionsView;

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

        pub fn populate_permutation_rows_flattened(
            interactions: DeviceInteractionsView<BabyBear>,
            permutation: MatrixViewMutDevice<BabyBear>,
            preprocessed: MatrixViewDevice<BabyBear>,
            main: MatrixViewDevice<BabyBear>,
            alpha: BinomialExtensionField<BabyBear, 4>,
            beta: BinomialExtensionField<BabyBear, 4>,
            batch_size: usize,
            num_blocks: usize,
            num_threads_per_block: usize,
        );
    }
}
