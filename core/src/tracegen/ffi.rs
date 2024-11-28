use crate::cuda_runtime::stream::CudaStreamHandle;
use crate::matrix::MatrixViewMutDevice;
use p3_baby_bear::BabyBear;
use sp1_core_executor::events::AluEvent;
use sp1_core_executor::events::MemoryInitializeFinalizeEvent;
use sp1_core_executor::events::MemoryLocalEvent;
use sp1_core_executor::events::SyscallEvent;
use sp1_recursion_core::{
    BaseAluEvent, BaseAluInstr, BatchFRIEvent, CommitPublicValuesEvent, ExtAluEvent, FriFoldEvent,
    Poseidon2Event, SelectEvent,
};
use sp1_stark::septic_curve::SepticCurve;

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

    pub fn core_memory_local_generate_trace_round_1(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const MemoryLocalEvent,
        nb_events: u32,
        stream: CudaStreamHandle,
    );

    pub fn core_memory_local_generate_trace_round_2(
        trace: MatrixViewMutDevice<BabyBear>,
        cumulative_sums: *mut SepticCurve<BabyBear>,
        stream: CudaStreamHandle,
    );

    pub fn core_memory_local_generate_trace_round_3(
        trace: MatrixViewMutDevice<BabyBear>,
        cumulative_sums: *const SepticCurve<BabyBear>,
        nb_events: u32,
        stream: CudaStreamHandle,
    );

    pub fn core_memory_global_generate_trace_round_1(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const MemoryInitializeFinalizeEvent,
        previous_addr: u32,
        nb_events: u32,
        is_receive: bool,
        stream: CudaStreamHandle,
    );

    pub fn core_memory_global_generate_trace_round_2(
        trace: MatrixViewMutDevice<BabyBear>,
        cumulative_sums: *mut SepticCurve<BabyBear>,
        stream: CudaStreamHandle,
    );

    pub fn core_memory_global_generate_trace_round_3(
        trace: MatrixViewMutDevice<BabyBear>,
        cumulative_sums: *const SepticCurve<BabyBear>,
        nb_events: u32,
        stream: CudaStreamHandle,
    );

    pub fn core_syscall_generate_trace_round_1(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const SyscallEvent,
        nb_events: u32,
        is_receive: bool,
        stream: CudaStreamHandle,
    );

    pub fn core_syscall_generate_trace_round_2(
        trace: MatrixViewMutDevice<BabyBear>,
        cumulative_sums: *mut SepticCurve<BabyBear>,
        stream: CudaStreamHandle,
    );

    pub fn core_syscall_generate_trace_round_3(
        trace: MatrixViewMutDevice<BabyBear>,
        cumulative_sums: *const SepticCurve<BabyBear>,
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
    pub fn recursion_base_alu_generate_preprocessed_trace(
        trace: MatrixViewMutDevice<BabyBear>,
        instructions: *const &BaseAluInstr<BabyBear>,
        nb_instructions: u32,
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

    // pub fn recursion_exp_reverse_bits_generate_trace(
    //     trace: MatrixViewMutDevice<BabyBear>,
    //     events: *const ExpReverseBitsEvent<BabyBear>,
    //     nb_events: u32,
    //     stream: CudaStreamHandle,
    // );

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

    pub fn recursion_poseidon2_skinny_generate_trace(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const Poseidon2Event<BabyBear>,
        nb_events: u32,
        stream: CudaStreamHandle,
    );

    pub fn recursion_poseidon2_wide_generate_trace(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const Poseidon2Event<BabyBear>,
        nb_events: u32,
        stream: CudaStreamHandle,
    );
}
