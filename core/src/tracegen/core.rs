use std::any::Any;

use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use sp1_core_executor::events::AluEvent;
use sp1_core_machine::alu::AddSubChip;
use sp1_core_machine::utils::next_power_of_two;
use sp1_stark::MachineTrace;

use crate::{
    cuda_runtime::{
        ffi::{cuda_stream_synchronize, DEFAULT_STREAM},
        stream::{CudaStream, CudaStreamHandle},
    },
    device::{error::CudaError, memory::ToDevice, DeviceBuffer},
    matrix::ColMajorMatrixDevice,
};

use super::DeviceAir;
use crate::tracegen;

impl DeviceAir<BabyBear> for AddSubChip {
    fn generate_trace_host(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Option<MachineTrace<BabyBear>> {
        let events = input
            .add_events
            .iter()
            .chain(input.sub_events.iter())
            .copied()
            .collect::<Vec<_>>()
            .to_device_async(stream)
            .unwrap();
        Some(MachineTrace::Deferred(Box::new(events)))
    }

    fn generate_trace_device(
        &self,
        input: Box<dyn Any + Send + Sync>,
        num_rows: usize,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        // Get the events.
        let events = input.downcast::<DeviceBuffer<AluEvent>>().unwrap();

        // Allocate the matrix.
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <AddSubChip as BaseAir<BabyBear>>::width(self),
            num_rows,
            stream,
        )?;

        // Generate the trace.
        unsafe {
            trace.set_max_width();
            tracegen::ffi::core_add_sub_generate_trace(
                trace.view_mut(),
                events.as_ptr(),
                events.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
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

    use crate::tracegen;

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
            tracegen::ffi::core_add_sub_generate_trace(
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
