use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

use crate::{
    cuda_runtime::stream::CudaStreamHandle,
    matrix::{MatrixViewDevice, MatrixViewMutDevice},
};

use super::DeviceInteractionsView;

extern "C" {
    pub fn populate_permutation_rows_flattened(
        interactions: DeviceInteractionsView<BabyBear>,
        permutation: MatrixViewMutDevice<BabyBear>,
        preprocessed: MatrixViewDevice<BabyBear>,
        main: MatrixViewDevice<BabyBear>,
        global_alpha: BinomialExtensionField<BabyBear, 4>,
        global_beta: BinomialExtensionField<BabyBear, 4>,
        local_alpha: BinomialExtensionField<BabyBear, 4>,
        local_beta: BinomialExtensionField<BabyBear, 4>,
        batch_size: usize,
        num_blocks: usize,
        num_threads_per_block: usize,
        stream: CudaStreamHandle,
    );
}

pub(super) mod quotient_gpu {
    use crate::{
        cuda_runtime::stream::CudaStreamHandle,
        matrix::{MatrixViewDevice, MatrixViewMutDevice},
        stark::quotient::TwoAdicMultiplicativeCosetDevice,
    };
    use air_v2::instruction::Instruction;
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    #[link_name = "quotient_gpu"]
    #[allow(unused_attributes)]
    extern "C" {
        #[link_name = "computeValues"]
        #[allow(unused)]
        pub fn compute_values(
            eval_program: *const Instruction,
            eval_program_len: usize,
            eval_constants: *const BinomialExtensionField<BabyBear, 4>,
            memory_size: usize,
            cumulative_sums: *const BinomialExtensionField<BabyBear, 4>,
            trace_domain: TwoAdicMultiplicativeCosetDevice<BabyBear>,
            quotient_domain: TwoAdicMultiplicativeCosetDevice<BabyBear>,
            preprocessed_trace_on_quotient_domain: MatrixViewDevice<BabyBear>,
            main_trace_on_quotient_domain: MatrixViewDevice<BabyBear>,
            permutation_trace_on_quotient_domain: MatrixViewDevice<BabyBear>,
            perm_challenges: *const BinomialExtensionField<BabyBear, 4>,
            alpha: BinomialExtensionField<BabyBear, 4>,
            public_values: *const BabyBear,
            trace_domain_generator: BabyBear,
            generator_powers: *const BabyBear,
            quotient_values: MatrixViewMutDevice<BabyBear>,
            num_blocks: usize,
            num_threads_per_block: usize,
            stream: CudaStreamHandle,
        );
    }
}
