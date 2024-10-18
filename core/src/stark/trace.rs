use p3_field::PrimeField32;
use sp1_core_executor::events::AluEvent;
use sp1_core_machine::riscv::RiscvAir;
use sp1_core_machine::CpuEventFfi;
use sp1_stark::air::MachineAir;

use crate::baby_bear::F;
use crate::cuda_runtime::stream::{CudaStream, CudaStreamHandle};
use crate::device::error::{CudaError, CudaRustError};
use crate::device::memory::ToDevice;
use crate::matrix::{ColMajorMatrixDevice, MatrixViewMutDevice};

pub trait AccelAir<T: PrimeField32>: MachineAir<T> {
    fn generate_trace_accel(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Result<ColMajorMatrixDevice<T>, CudaError>;
}

impl AccelAir<F> for RiscvAir<F> {
    fn generate_trace_accel(
        &self,
        input: &Self::Record,
        output_record: &mut Self::Record,
    ) -> Result<ColMajorMatrixDevice<F>, CudaError> {
        match self {
            // RiscvAir::Cpu(_) => cpu_generate_trace(
            //     // Eventually, we'll make CPU events FFI compatible.
            //     &input
            //         .cpu_events
            //         .iter()
            //         .map(|event| CpuEventFfi::new(event, &input.nonce_lookup))
            //         .collect::<Vec<_>>(),
            // ),
            RiscvAir::Add(_) => add_sub_generate_trace(
                // &[&input.add_events, &input.sub_events]
                //     .into_iter()
                //     .flatten()
                //     .cloned()
                //     .collect::<Vec<_>>(),
                &input.add_events, // Ignore sub_events for now. Should be combined later.
            ),
            // RiscvAir::Bitwise(_) => bitwise_generate_trace(&input.bitwise_events),
            // RiscvAir::Lt(_) => lt_generate_trace(&input.lt_events),
            // RiscvAir::ShiftLeft(_) => sll_generate_trace(&input.shift_left_events),
            // RiscvAir::ShiftRight(_) => sr_generate_trace(&input.shift_right_events),
            // Fallback for other chips.
            other => {
                let mat = other.generate_trace(input, output_record).to_device()?.to_column_major();
                mat.stream().synchronize()?;
                Ok(mat)
            }
        }
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

pub fn add_sub_generate_trace(events: &[AluEvent]) -> Result<ColMajorMatrixDevice<F>, CudaError> {
    const NUM_COLS: usize = sp1_core_machine::alu::NUM_ADD_SUB_COLS;
    const ROWS_PER_EVENT: usize = 1;

    let nb_rows = (events.len() * ROWS_PER_EVENT).next_power_of_two();

    // let stream = CudaStream::create()?;
    let stream = CudaStream::default();
    let add_events = events.to_device_async(&stream)?;
    let mut mat = ColMajorMatrixDevice::<F>::with_capacity_in(NUM_COLS, nb_rows, &stream)?;
    unsafe { mat.values.set_max_len() };

    unsafe {
        add_sub_populate_babybear(
            mat.view_mut(),
            add_events.as_ptr(),
            add_events.len(),
            stream.handle(),
        )
    }
    .to_result()?;

    Ok(mat)
}

// extern "C" {
//     pub fn bitwise_populate_babybear(
//         mat: MatrixViewMutDevice<F>,
//         events: *const AluEvent,
//         nb_events: usize,
//         stream: CudaStreamHandle,
//     ) -> CudaRustError;
// }

// pub fn bitwise_generate_trace(events: &[AluEvent]) -> Result<ColMajorMatrixDevice<F>, CudaError> {
//     const NUM_COLS: usize = sp1_core_machine::alu::bitwise::NUM_BITWISE_COLS;
//     const ROWS_PER_EVENT: usize = 1;

//     let nb_rows = (events.len() * ROWS_PER_EVENT).next_power_of_two();

//     let add_events = events.to_device()?;
//     let mut mat = ColMajorMatrixDevice::<F>::with_capacity(NUM_COLS, nb_rows)?;
//     unsafe { mat.values.set_max_len() };

//     unsafe {
//         bitwise_populate_babybear(
//             mat.view_mut(),
//             add_events.as_ptr(),
//             add_events.len(),
//             mat.stream().handle(),
//         )
//     }
//     .to_result()?;

//     Ok(mat)
// }

// extern "C" {
//     pub fn lt_populate_babybear(
//         mat: MatrixViewMutDevice<F>,
//         events: *const AluEvent,
//         nb_events: usize,
//         stream: CudaStreamHandle,
//     ) -> CudaRustError;
// }

// pub fn lt_generate_trace(events: &[AluEvent]) -> Result<ColMajorMatrixDevice<F>, CudaError> {
//     const NUM_COLS: usize = sp1_core_machine::alu::lt::NUM_LT_COLS;
//     const ROWS_PER_EVENT: usize = 1;

//     let nb_rows = (events.len() * ROWS_PER_EVENT).next_power_of_two();

//     let add_events = events.to_device()?;
//     let mut mat = ColMajorMatrixDevice::<F>::with_capacity(NUM_COLS, nb_rows)?;
//     unsafe { mat.values.set_max_len() };

//     unsafe {
//         lt_populate_babybear(
//             mat.view_mut(),
//             add_events.as_ptr(),
//             add_events.len(),
//             mat.stream().handle(),
//         )
//     }
//     .to_result()?;

//     Ok(mat)
// }

// extern "C" {
//     pub fn sll_populate_babybear(
//         mat: MatrixViewMutDevice<F>,
//         events: *const AluEvent,
//         nb_events: usize,
//         stream: CudaStreamHandle,
//     ) -> CudaRustError;
// }

// pub fn sll_generate_trace(events: &[AluEvent]) -> Result<ColMajorMatrixDevice<F>, CudaError> {
//     const NUM_COLS: usize = sp1_core_machine::alu::sll::NUM_SHIFT_LEFT_COLS;
//     const ROWS_PER_EVENT: usize = 1;

//     let nb_rows = (events.len() * ROWS_PER_EVENT).next_power_of_two();

//     let add_events = events.to_device()?;
//     let mut mat = ColMajorMatrixDevice::<F>::with_capacity(NUM_COLS, nb_rows)?;
//     unsafe { mat.values.set_max_len() };

//     unsafe {
//         sll_populate_babybear(
//             mat.view_mut(),
//             add_events.as_ptr(),
//             add_events.len(),
//             mat.stream().handle(),
//         )
//     }
//     .to_result()?;

//     Ok(mat)
// }

// extern "C" {
//     pub fn sr_populate_babybear(
//         mat: MatrixViewMutDevice<F>,
//         events: *const AluEvent,
//         nb_events: usize,
//         stream: CudaStreamHandle,
//     ) -> CudaRustError;
// }

// pub fn sr_generate_trace(events: &[AluEvent]) -> Result<ColMajorMatrixDevice<F>, CudaError> {
//     const NUM_COLS: usize = sp1_core_machine::alu::sr::NUM_SHIFT_RIGHT_COLS;
//     const ROWS_PER_EVENT: usize = 1;

//     let nb_rows = (events.len() * ROWS_PER_EVENT).next_power_of_two();

//     let add_events = events.to_device()?;
//     let mut mat = ColMajorMatrixDevice::<F>::with_capacity(NUM_COLS, nb_rows)?;
//     unsafe { mat.values.set_max_len() };

//     unsafe {
//         sr_populate_babybear(
//             mat.view_mut(),
//             add_events.as_ptr(),
//             add_events.len(),
//             mat.stream().handle(),
//         )
//     }
//     .to_result()?;

//     Ok(mat)
// }

// extern "C" {
//     pub fn cpu_populate_babybear(
//         mat: MatrixViewMutDevice<F>,
//         events: *const CpuEventFfi,
//         nb_events: usize,
//         stream: CudaStreamHandle,
//     ) -> CudaRustError;
// }

// pub fn cpu_generate_trace(events: &[CpuEventFfi]) -> Result<ColMajorMatrixDevice<F>, CudaError> {
//     const NUM_COLS: usize = sp1_core_machine::cpu::columns::NUM_CPU_COLS;
//     const ROWS_PER_EVENT: usize = 1;

//     let nb_rows = (events.len() * ROWS_PER_EVENT).next_power_of_two();

//     let add_events = events.to_device()?;
//     let mut mat = ColMajorMatrixDevice::<F>::with_capacity(NUM_COLS, nb_rows)?;
//     unsafe { mat.values.set_max_len() };

//     unsafe {
//         cpu_populate_babybear(
//             mat.view_mut(),
//             add_events.as_ptr(),
//             add_events.len(),
//             mat.stream().handle(),
//         )
//     }
//     .to_result()?;

//     Ok(mat)
// }
