mod permutation;

pub use permutation::*;

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
    }
}
