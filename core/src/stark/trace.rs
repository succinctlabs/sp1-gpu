use p3_field::PrimeField32;
use rayon::prelude::*;
use sp1_core_executor::events::AluEvent;
use sp1_core_executor::ExecutionRecord;
use sp1_core_machine::alu::{AddSubChip, BitwiseChip, LtChip, ShiftLeftChip, ShiftRightChip};
use sp1_core_machine::cpu::CpuChip;
use sp1_core_machine::riscv::RiscvAir;
use sp1_core_machine::sys::CpuEventFfi;
use sp1_core_machine::utils::next_power_of_two;
use sp1_recursion_core::machine::RecursionAir;
use sp1_stark::air::MachineAir;

use crate::baby_bear::F;
use crate::cuda_runtime::stream::{CudaStream, CudaStreamHandle};
use crate::device::error::{CudaError, CudaRustError};
use crate::device::memory::ToDevice;
use crate::matrix::{ColMajorMatrixDevice, MatrixViewMutDevice};

/// An AIR that possibly has hardware accelerated functionality.
pub trait AccelAir<F: PrimeField32>: MachineAir<F> {
    /// Generate a trace, using hardware acceleration if available.
    fn generate_trace_accel(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<ColMajorMatrixDevice<F>, CudaError>;
}

impl AccelAir<F> for RiscvAir<F> {
    fn generate_trace_accel(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<ColMajorMatrixDevice<F>, CudaError> {
        // Accelerate trace generation for available chips.
        match self {
            RiscvAir::Cpu(chip) => chip.generate_trace_ffi(input, output, stream),
            RiscvAir::Add(chip) => chip.generate_trace_ffi(input, output, stream),
            RiscvAir::Bitwise(chip) => chip.generate_trace_ffi(input, output, stream),
            RiscvAir::Lt(chip) => chip.generate_trace_ffi(input, output, stream),
            RiscvAir::ShiftLeft(chip) => chip.generate_trace_ffi(input, output, stream),
            RiscvAir::ShiftRight(chip) => chip.generate_trace_ffi(input, output, stream),
            // Fallback for other chips.
            other => tracing::debug_span!("on host").in_scope(|| {
                let trace = tracing::debug_span!("generate")
                    .in_scope(|| other.generate_trace(input, output));
                tracing::debug_span!("to device")
                    .in_scope(|| Ok(trace.to_device_async(stream)?.to_column_major()))
            }),
        }
    }
}

impl<const DEGREE: usize> AccelAir<F> for RecursionAir<F, DEGREE> {
    fn generate_trace_accel(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<ColMajorMatrixDevice<F>, CudaError> {
        Ok(self.generate_trace(input, output).to_device_async(stream)?.to_column_major())
    }
}

/// An AIR that has functionality available through FFI.
pub trait FfiAir: MachineAir<F> {
    const NUM_COLS: usize;
    const ROWS_PER_EVENT: usize = 1;
    const FFI_POPULATE: FfiPopulate<Self::Event>;
    type Event: Copy;

    fn events(input: &ExecutionRecord) -> impl AsRef<[Self::Event]>;

    fn generate_trace_ffi(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
        stream: &CudaStream,
    ) -> Result<ColMajorMatrixDevice<F>, CudaError> {
        // These two vectors should be combined in the record struct.
        let events_owned = Self::events(input);
        let events = events_owned.as_ref();

        let nb_rows = next_power_of_two(
            events.len() * Self::ROWS_PER_EVENT,
            input.fixed_log2_rows::<F, _>(self),
        );

        let events = events.to_device_async(stream)?;
        let mut mat = ColMajorMatrixDevice::<F>::with_capacity_in(Self::NUM_COLS, nb_rows, stream)?;
        unsafe { mat.set_max_width() };

        unsafe {
            Self::FFI_POPULATE(mat.view_mut(), events.as_ptr(), events.len(), stream.handle())
        }
        .to_result()?;

        Ok(mat)
    }
}

/// The type of an FFI populate function associated with an `Event` type.
type FfiPopulate<Event> = unsafe extern "C" fn(
    mat: MatrixViewMutDevice<F>,
    events: *const Event,
    nb_events: usize,
    stream: CudaStreamHandle,
) -> CudaRustError;

// These `extern` declarations cannot be factored through a macro, because cbindgen needs
// to read them directly. It does have functionality to expand macros, but this functionality
// seems to be more trouble than it is worth.

extern "C" {
    pub fn add_sub_populate_babybear(
        mat: MatrixViewMutDevice<F>,
        events: *const AluEvent,
        nb_events: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

impl FfiAir for AddSubChip {
    const NUM_COLS: usize = sp1_core_machine::alu::NUM_ADD_SUB_COLS;
    const FFI_POPULATE: FfiPopulate<Self::Event> = add_sub_populate_babybear;
    type Event = AluEvent;

    fn events(input: &ExecutionRecord) -> impl AsRef<[Self::Event]> {
        &input.add_sub_events
    }
}

extern "C" {
    pub fn bitwise_populate_babybear(
        mat: MatrixViewMutDevice<F>,
        events: *const AluEvent,
        nb_events: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

impl FfiAir for BitwiseChip {
    const NUM_COLS: usize = sp1_core_machine::alu::bitwise::NUM_BITWISE_COLS;
    const FFI_POPULATE: FfiPopulate<Self::Event> = bitwise_populate_babybear;
    type Event = AluEvent;

    fn events(input: &ExecutionRecord) -> impl AsRef<[Self::Event]> {
        &input.bitwise_events
    }
}

extern "C" {
    pub fn lt_populate_babybear(
        mat: MatrixViewMutDevice<F>,
        events: *const AluEvent,
        nb_events: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

impl FfiAir for LtChip {
    const NUM_COLS: usize = sp1_core_machine::alu::lt::NUM_LT_COLS;
    const FFI_POPULATE: FfiPopulate<Self::Event> = lt_populate_babybear;
    type Event = AluEvent;

    fn events(input: &ExecutionRecord) -> impl AsRef<[Self::Event]> {
        &input.lt_events
    }
}

extern "C" {
    pub fn sll_populate_babybear(
        mat: MatrixViewMutDevice<F>,
        events: *const AluEvent,
        nb_events: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

impl FfiAir for ShiftLeftChip {
    const NUM_COLS: usize = sp1_core_machine::alu::sll::NUM_SHIFT_LEFT_COLS;
    const FFI_POPULATE: FfiPopulate<Self::Event> = sll_populate_babybear;
    type Event = AluEvent;

    fn events(input: &ExecutionRecord) -> impl AsRef<[Self::Event]> {
        &input.shift_left_events
    }
}

extern "C" {
    pub fn sr_populate_babybear(
        mat: MatrixViewMutDevice<F>,
        events: *const AluEvent,
        nb_events: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

impl FfiAir for ShiftRightChip {
    const NUM_COLS: usize = sp1_core_machine::alu::sr::NUM_SHIFT_RIGHT_COLS;
    const FFI_POPULATE: FfiPopulate<Self::Event> = sr_populate_babybear;
    type Event = AluEvent;

    fn events(input: &ExecutionRecord) -> impl AsRef<[Self::Event]> {
        &input.shift_right_events
    }
}

extern "C" {
    pub fn cpu_populate_babybear(
        mat: MatrixViewMutDevice<F>,
        events: *const CpuEventFfi,
        nb_events: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

impl FfiAir for CpuChip {
    const NUM_COLS: usize = sp1_core_machine::cpu::columns::NUM_CPU_COLS;
    const FFI_POPULATE: FfiPopulate<Self::Event> = cpu_populate_babybear;
    type Event = CpuEventFfi;

    fn events(input: &ExecutionRecord) -> impl AsRef<[Self::Event]> {
        tracing::debug_span!("cpu events translation").in_scope(|| {
            input
                .cpu_events
                .par_iter()
                .map(|event| CpuEventFfi::new(event, &input.nonce_lookup))
                .collect::<Vec<_>>()
        })
    }
}

#[cfg(test)]
mod tests {
    use rayon::prelude::*;

    use sp1_core_executor::{ExecutionRecord, Executor, Program};
    use sp1_core_machine::riscv::RiscvAir;
    use sp1_stark::{air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, SP1CoreOpts};

    use crate::{cuda_runtime::stream::CudaStream, device::memory::ToHost, utils::init_tracer};

    use super::AccelAir;

    const FIBONACCI_ELF: &[u8] =
        include_bytes!("../../../perf/programs/fibonacci/riscv32im-succinct-zkvm-elf");

    #[test]
    fn generate_trace_gpu_eq_cpu() {
        init_tracer();
        let config = BabyBearPoseidon2::new();
        let machine = RiscvAir::machine(config);

        let program = Program::from(FIBONACCI_ELF).unwrap();
        let mut executor = Executor::new(program, SP1CoreOpts::default());
        executor.run().unwrap();
        let records = &executor.records;

        for record in records {
            let traces = machine
                .chips()
                .par_iter()
                .filter(|chip| chip.included(record))
                .map(|chip| {
                    let mat = chip
                        .air
                        .generate_trace_accel(
                            record,
                            &mut ExecutionRecord::default(),
                            &CudaStream::default(),
                        )
                        .unwrap();

                    let trace_cpu = chip.generate_trace(record, &mut ExecutionRecord::default());

                    mat.stream().synchronize().unwrap();
                    let trace_gpu = mat.to_host();

                    (chip.name(), trace_gpu, trace_cpu)
                })
                .collect::<Vec<_>>();

            for (name, trace_gpu, trace_cpu) in traces {
                assert_eq!(
                    trace_gpu, trace_cpu,
                    "chip {name}'s gpu trace should be the same as cpu trace"
                );
            }
        }
    }
}
