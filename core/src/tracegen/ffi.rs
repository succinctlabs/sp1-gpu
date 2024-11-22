use crate::cuda_runtime::stream::CudaStreamHandle;
use crate::matrix::MatrixViewMutDevice;
use p3_baby_bear::BabyBear;
use sp1_core_executor::events::AluEvent;
use sp1_recursion_core::{
    BaseAluEvent, BatchFRIEvent, CommitPublicValuesEvent, ExpReverseBitsEventFFI, ExtAluEvent,
    FriFoldEvent, SelectEvent,
};

/// cbindgen:ignore
#[allow(unused_attributes)]
#[link_name = "moongate"]
extern "C" {
    pub fn core_add_sub_generate_trace(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const AluEvent,
        nb_events: u32,
        stream: CudaStreamHandle,
    );
}

/// cbindgen:ignore
#[allow(unused_attributes)]
#[link_name = "moongate"]
extern "C" {
    pub fn recursion_base_alu_generate_trace(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const BaseAluEvent<BabyBear>,
        nb_events: u32,
        stream: CudaStreamHandle,
    );

    pub fn recursion_ext_alu_generate_trace(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const ExtAluEvent<BabyBear>,
        nb_events: u32,
        stream: CudaStreamHandle,
    );

    pub fn recursion_batch_fri_generate_trace(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const BatchFRIEvent<BabyBear>,
        nb_events: u32,
        stream: CudaStreamHandle,
    );

    pub fn recursion_exp_reverse_bits_generate_trace(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const ExpReverseBitsEventFFI<BabyBear>,
        nb_events: u32,
        stream: CudaStreamHandle,
    );

    pub fn recursion_fri_fold_generate_trace(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const FriFoldEvent<BabyBear>,
        nb_events: u32,
        stream: CudaStreamHandle,
    );

    pub fn recursion_public_values_generate_trace(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const CommitPublicValuesEvent<BabyBear>,
        nb_events: u32,
        stream: CudaStreamHandle,
    );

    pub fn recursion_select_generate_trace(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const SelectEvent<BabyBear>,
        nb_events: u32,
        stream: CudaStreamHandle,
    );
}
