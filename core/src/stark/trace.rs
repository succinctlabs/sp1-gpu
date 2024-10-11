use sp1_core_executor::events::AluEvent;
use sp1_core_machine::alu::NUM_ADD_SUB_COLS;

use crate::baby_bear::F;
use crate::device::error::{CudaError, CudaRustError};
use crate::device::memory::ToDevice;
use crate::matrix::{ColMajorMatrixDevice, MatrixViewMutDevice, RowMajorMatrixDevice};

extern "C" {
    pub fn add_sub_events_to_rows_babybear(
        mat: MatrixViewMutDevice<F>,
        events: *const AluEvent,
        nb_events: usize,
    ) -> CudaRustError;
}

pub fn add_sub_generate_trace(
    add_events: &[AluEvent],
) -> Result<ColMajorMatrixDevice<F>, CudaError> {
    const NUM_COLS: usize = NUM_ADD_SUB_COLS;
    const ROWS_PER_EVENT: usize = 1;

    let nb_rows = (add_events.len() * ROWS_PER_EVENT).next_power_of_two();

    // let x = unsafe { CudaStream::default().alloc(NUM_COLS * nb_rows * size_of::<F>())? };
    let add_events = add_events.to_device()?;
    let mut mat = RowMajorMatrixDevice::<F>::with_capacity(NUM_COLS, nb_rows)?;
    unsafe { mat.values.set_max_len() };
    // let mut mat = RowMajorMatrixDevice::<F>::new(
    //     vec![F::default(); NUM_COLS * nb_rows].to_device()?,
    //     NUM_COLS,
    // );

    unsafe {
        add_sub_events_to_rows_babybear(mat.view_mut(), add_events.as_ptr(), add_events.len())
    }
    .to_result()?;

    // mat.stream().synchronize().unwrap();
    // // println!("{:?}", mat.to_host());

    Ok(mat.to_column_major())
}
