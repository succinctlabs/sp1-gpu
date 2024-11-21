use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use sp1_core_machine::alu::AddSubChip;
use sp1_core_machine::utils::next_power_of_two;

use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaError, memory::ToDevice},
    matrix::ColMajorMatrixDevice,
};

use super::ffi::add_sub;
use super::DeviceAir;

impl DeviceAir<BabyBear> for AddSubChip {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        // Get the events for the chip.
        let events =
            input.add_events.iter().chain(input.sub_events.iter()).copied().collect::<Vec<_>>();

        // Copy the events to device.
        let events = events.to_device_async(stream)?;

        // Get the number of rows.
        let nb_rows = next_power_of_two(
            input.add_events.len() + input.sub_events.len(),
            input.fixed_log2_rows::<BabyBear, _>(self),
        );

        // Allocate the matrix.
        let mut mat = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <AddSubChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        // Generate the trace.
        unsafe {
            mat.set_max_width();
            add_sub::generate_trace(
                mat.view_mut(),
                events.as_ptr(),
                events.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(mat))
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_power_of_two(
            input.add_events.len() + input.sub_events.len(),
            input.fixed_log2_rows::<BabyBear, _>(self),
        );
        Some(nb_rows)
    }
}

#[cfg(test)]
mod tests {
    use crate::device::memory::ToHost;
    use crate::{
        cuda_runtime::ffi::DEFAULT_STREAM, device::memory::ToDevice, matrix::RowMajorMatrixDevice,
    };
    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use sp1_core_executor::{events::AluEvent, ExecutionRecord, Opcode};
    use sp1_core_machine::alu::AddSubChip;
    use sp1_stark::air::MachineAir;

    use super::add_sub;

    #[test]
    fn test_add_sub_generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.add_events = [AluEvent::new(0, 0, Opcode::ADD, 14, 8, 6)].repeat(100);

        let chip = AddSubChip;
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let mut trace_device =
            RowMajorMatrixDevice::new(trace.values.to_device().unwrap(), trace.width())
                .to_column_major();

        let events = shard.add_events.to_device().unwrap().as_ptr();
        unsafe {
            add_sub::generate_trace(
                trace_device.view_mut(),
                events,
                shard.add_events.len() as u32,
                DEFAULT_STREAM,
            );
        }

        let gpu_trace = trace_device.to_host();
        assert_eq!(trace, gpu_trace);
    }
}
