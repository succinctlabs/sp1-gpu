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

pub trait AccelAir<F: PrimeField32>: MachineAir<F> {
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
        match self {
            RiscvAir::Cpu(chip) => cpu_generate_trace(chip, input, output, stream),
            RiscvAir::Add(chip) => add_sub_generate_trace(chip, input, output, stream),
            RiscvAir::Bitwise(chip) => bitwise_generate_trace(chip, input, output, stream),
            RiscvAir::Lt(chip) => lt_generate_trace(chip, input, output, stream),
            RiscvAir::ShiftLeft(chip) => sll_generate_trace(chip, input, output, stream),
            RiscvAir::ShiftRight(chip) => sr_generate_trace(chip, input, output, stream),
            // Fallback for other chips.
            other => Ok(other.generate_trace(input, output).to_device()?.to_column_major()),
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
        let mat = self.generate_trace(input, output).to_device_async(stream)?.to_column_major();
        // mat.stream().synchronize()?;
        Ok(mat)
    }
}

// TODO: investigate if a macro can be used here.

extern "C" {
    pub fn add_sub_populate_babybear(
        mat: MatrixViewMutDevice<F>,
        events: *const AluEvent,
        nb_events: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

pub fn add_sub_generate_trace(
    chip: &AddSubChip,
    input: &ExecutionRecord,
    _output: &mut ExecutionRecord,
    stream: &CudaStream,
) -> Result<ColMajorMatrixDevice<F>, CudaError> {
    const NUM_COLS: usize = sp1_core_machine::alu::NUM_ADD_SUB_COLS;
    const ROWS_PER_EVENT: usize = 1;
    // These two vectors should be combined in the record struct.
    let events = &[&input.add_events, &input.sub_events]
        .into_par_iter()
        .flatten()
        .cloned()
        .collect::<Vec<_>>();

    let nb_rows =
        next_power_of_two(events.len() * ROWS_PER_EVENT, input.fixed_log2_rows::<F, _>(chip));

    let events = events.to_device_async(stream)?;
    let mut mat = ColMajorMatrixDevice::<F>::with_capacity_in(NUM_COLS, nb_rows, stream)?;
    unsafe { mat.set_max_width() };

    unsafe {
        add_sub_populate_babybear(mat.view_mut(), events.as_ptr(), events.len(), stream.handle())
    }
    .to_result()?;

    Ok(mat)
}

extern "C" {
    pub fn bitwise_populate_babybear(
        mat: MatrixViewMutDevice<F>,
        events: *const AluEvent,
        nb_events: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

pub fn bitwise_generate_trace(
    chip: &BitwiseChip,
    input: &ExecutionRecord,
    _output: &mut ExecutionRecord,
    stream: &CudaStream,
) -> Result<ColMajorMatrixDevice<F>, CudaError> {
    const NUM_COLS: usize = sp1_core_machine::alu::bitwise::NUM_BITWISE_COLS;
    const ROWS_PER_EVENT: usize = 1;
    let events = &input.bitwise_events;

    let nb_rows =
        next_power_of_two(events.len() * ROWS_PER_EVENT, input.fixed_log2_rows::<F, _>(chip));

    let events = events.to_device_async(stream)?;
    let mut mat = ColMajorMatrixDevice::<F>::with_capacity_in(NUM_COLS, nb_rows, stream)?;
    unsafe { mat.set_max_width() };

    unsafe {
        bitwise_populate_babybear(mat.view_mut(), events.as_ptr(), events.len(), stream.handle())
    }
    .to_result()?;

    Ok(mat)
}

extern "C" {
    pub fn lt_populate_babybear(
        mat: MatrixViewMutDevice<F>,
        events: *const AluEvent,
        nb_events: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

pub fn lt_generate_trace(
    chip: &LtChip,
    input: &ExecutionRecord,
    _output: &mut ExecutionRecord,
    stream: &CudaStream,
) -> Result<ColMajorMatrixDevice<F>, CudaError> {
    const NUM_COLS: usize = sp1_core_machine::alu::lt::NUM_LT_COLS;
    const ROWS_PER_EVENT: usize = 1;
    let events = &input.lt_events;

    let nb_rows =
        next_power_of_two(events.len() * ROWS_PER_EVENT, input.fixed_log2_rows::<F, _>(chip));

    let events = events.to_device_async(stream)?;
    let mut mat = ColMajorMatrixDevice::<F>::with_capacity_in(NUM_COLS, nb_rows, stream)?;
    unsafe { mat.set_max_width() };

    unsafe { lt_populate_babybear(mat.view_mut(), events.as_ptr(), events.len(), stream.handle()) }
        .to_result()?;

    Ok(mat)
}

extern "C" {
    pub fn sll_populate_babybear(
        mat: MatrixViewMutDevice<F>,
        events: *const AluEvent,
        nb_events: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

pub fn sll_generate_trace(
    chip: &ShiftLeftChip,
    input: &ExecutionRecord,
    _output: &mut ExecutionRecord,
    stream: &CudaStream,
) -> Result<ColMajorMatrixDevice<F>, CudaError> {
    const NUM_COLS: usize = sp1_core_machine::alu::sll::NUM_SHIFT_LEFT_COLS;
    const ROWS_PER_EVENT: usize = 1;
    let events = &input.shift_left_events;

    let nb_rows =
        next_power_of_two(events.len() * ROWS_PER_EVENT, input.fixed_log2_rows::<F, _>(chip));

    let events = events.to_device_async(stream)?;
    let mut mat = ColMajorMatrixDevice::<F>::with_capacity_in(NUM_COLS, nb_rows, stream)?;
    unsafe { mat.set_max_width() };

    unsafe {
        sll_populate_babybear(mat.view_mut(), events.as_ptr(), events.len(), stream.handle())
    }
    .to_result()?;

    Ok(mat)
}

extern "C" {
    pub fn sr_populate_babybear(
        mat: MatrixViewMutDevice<F>,
        events: *const AluEvent,
        nb_events: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

pub fn sr_generate_trace(
    chip: &ShiftRightChip,
    input: &ExecutionRecord,
    _output: &mut ExecutionRecord,
    stream: &CudaStream,
) -> Result<ColMajorMatrixDevice<F>, CudaError> {
    const NUM_COLS: usize = sp1_core_machine::alu::sr::NUM_SHIFT_RIGHT_COLS;
    const ROWS_PER_EVENT: usize = 1;
    let events = &input.shift_right_events;

    let nb_rows =
        next_power_of_two(events.len() * ROWS_PER_EVENT, input.fixed_log2_rows::<F, _>(chip));

    let events = events.to_device_async(stream)?;
    let mut mat = ColMajorMatrixDevice::<F>::with_capacity_in(NUM_COLS, nb_rows, stream)?;
    unsafe { mat.set_max_width() };

    unsafe { sr_populate_babybear(mat.view_mut(), events.as_ptr(), events.len(), stream.handle()) }
        .to_result()?;

    Ok(mat)
}

extern "C" {
    pub fn cpu_populate_babybear(
        mat: MatrixViewMutDevice<F>,
        events: *const CpuEventFfi,
        nb_events: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

pub fn cpu_generate_trace(
    chip: &CpuChip,
    input: &ExecutionRecord,
    _output: &mut ExecutionRecord,
    stream: &CudaStream,
) -> Result<ColMajorMatrixDevice<F>, CudaError> {
    const NUM_COLS: usize = sp1_core_machine::cpu::columns::NUM_CPU_COLS;
    const ROWS_PER_EVENT: usize = 1;
    // Eventually, we'll make CPU events FFI compatible.
    let events = &input
        .cpu_events
        .par_iter()
        .map(|event| CpuEventFfi::new(event, &input.nonce_lookup))
        .collect::<Vec<_>>();

    let nb_rows =
        next_power_of_two(events.len() * ROWS_PER_EVENT, input.fixed_log2_rows::<F, _>(chip));

    let events = events.to_device_async(stream)?;
    let mut mat = ColMajorMatrixDevice::<F>::with_capacity_in(NUM_COLS, nb_rows, stream)?;
    unsafe { mat.set_max_width() };

    unsafe {
        cpu_populate_babybear(mat.view_mut(), events.as_ptr(), events.len(), stream.handle())
    }
    .to_result()?;

    Ok(mat)
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
