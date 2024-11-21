use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use sp1_core_machine::utils::next_power_of_two;
use sp1_recursion_core::chips::alu_base::{BaseAluChip, NUM_BASE_ALU_ENTRIES_PER_ROW};

use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaError, memory::ToDevice},
    matrix::ColMajorMatrixDevice,
};

use super::DeviceAir;
use crate::tracegen;

impl DeviceAir<BabyBear> for BaseAluChip {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        // Get the events for the chip.
        let events = &input.base_alu_events;

        // Copy the events to device.
        let events = events.to_device_async(stream)?;

        // Get the number of rows.
        let nb_rows = self.num_rows(input).unwrap();

        // Allocate the matrix.
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <BaseAluChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        // Generate the trace.
        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_base_alu_generate_trace(
                trace.view_mut(),
                events.as_ptr(),
                events.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.base_alu_events;
        Some(next_power_of_two(
            events.len().div_ceil(NUM_BASE_ALU_ENTRIES_PER_ROW),
            input.fixed_log2_rows(self),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cuda_runtime::stream::CudaStream, device::memory::ToDevice, matrix::RowMajorMatrixDevice,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use sp1_recursion_core::chips::alu_base::BaseAluChip;
    use sp1_recursion_core::{BaseAluIo, ExecutionRecord};
    use sp1_stark::air::MachineAir;

    use crate::tracegen::DeviceAir;

    #[test]
    fn test_recursion_base_alu_generate_trace() {
        type F = BabyBear;

        let chip = BaseAluChip;
        let shard = ExecutionRecord {
            base_alu_events: vec![BaseAluIo { out: F::one(), in1: F::one(), in2: F::one() }],
            ..Default::default()
        };
        let trace: RowMajorMatrix<F> = chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let device_trace = chip
            .generate_trace_device(&shard, &mut ExecutionRecord::default(), &CudaStream::default())
            .unwrap()
            .unwrap();
        assert_eq!(trace, device_trace.to_host_naive());
    }
}
