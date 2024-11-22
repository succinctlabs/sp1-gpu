use crate::cuda_runtime::stream::CudaStreamHandle;
use crate::matrix::MatrixViewMutDevice;
use p3_baby_bear::BabyBear;
use sp1_core_executor::events::AluEvent;
use sp1_core_executor::events::MemoryInitializeFinalizeEvent;
use sp1_core_executor::events::MemoryLocalEvent;
use sp1_core_executor::events::SyscallEvent;
use sp1_recursion_core::BaseAluEvent;
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
}
