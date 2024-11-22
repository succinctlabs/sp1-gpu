use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use sp1_core_machine::utils::next_power_of_two;
use sp1_core_machine::{alu::AddSubChip, memory::MemoryLocalChip};

use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaError, memory::ToDevice},
    matrix::ColMajorMatrixDevice,
};

use super::DeviceAir;
use crate::tracegen;

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
        let nb_rows = self.num_rows(input).unwrap();

        // Allocate the matrix.
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <AddSubChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
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

impl DeviceAir<BabyBear> for MemoryLocalChip {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        // Get the events for the chip.
        let events = input.get_local_mem_events().cloned().collect::<Vec<_>>();

        // Copy the events to device.
        let events = events.to_device_async(stream)?;

        // Get the number of rows.
        let nb_rows = self.num_rows(input).unwrap();

        // Allocate the matrix.
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <MemoryLocalChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        // Generate the trace.
        unsafe {
            trace.set_max_width();
            tracegen::ffi::core_memory_local_generate_trace_round_1(
                trace.view_mut(),
                events.as_ptr(),
                events.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = input.get_local_mem_events().collect::<Vec<_>>();
        let nb_rows = (events.len() + 3) / 4;
        let size_log2 = input.fixed_log2_rows::<BabyBear, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);
        Some(padded_nb_rows)
    }
}

#[cfg(test)]
mod tests {
    use crate::cuda_runtime::ffi::cuda_device_synchronize;
    use crate::cuda_runtime::stream::CudaStream;
    use crate::device::memory::ToHost;
    use crate::tracegen::ffi::core_memory_local_generate_trace_round_2;
    use crate::{
        cuda_runtime::ffi::DEFAULT_STREAM, device::memory::ToDevice, matrix::RowMajorMatrixDevice,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use sp1_core_executor::events::MemoryRecord;
    use sp1_core_executor::{events::AluEvent, events::MemoryLocalEvent, ExecutionRecord, Opcode};
    use sp1_core_machine::alu::AddSubChip;
    use sp1_core_machine::memory::MemoryLocalChip;
    use sp1_stark::air::MachineAir;
    use sp1_stark::septic_curve::SepticCurve;

    use crate::tracegen;
    use rand::Rng;

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

    #[test]
    fn test_memory_local_generate_trace() {
        let mut rng = rand::thread_rng();
        let mut shard = ExecutionRecord::default();
        shard.cpu_local_memory_access = (0..2_700_000)
            .map(|_| MemoryLocalEvent {
                addr: rng.gen_range(0..10000),
                initial_mem_access: MemoryRecord {
                    shard: rng.gen_range(0..10000),
                    timestamp: rng.gen_range(0..10000),
                    value: rng.gen_range(0..10000),
                },
                final_mem_access: MemoryRecord {
                    shard: rng.gen_range(0..10000),
                    timestamp: rng.gen_range(0..10000),
                    value: rng.gen_range(0..10000),
                },
            })
            .collect::<Vec<_>>();

        let chip = MemoryLocalChip::new();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let mut trace_copy = trace.clone();
        trace_copy.values.fill(BabyBear::zero());
        let mut trace_device =
            RowMajorMatrixDevice::new(trace_copy.values.to_device().unwrap(), trace.width())
                .to_column_major();

        let events = shard.get_local_mem_events().cloned().collect::<Vec<_>>();
        let nb_events = events.len();
        let events = events.to_device().unwrap().as_ptr();
        unsafe {
            tracegen::ffi::core_memory_local_generate_trace_round_1(
                trace_device.view_mut(),
                events,
                nb_events as u32,
                DEFAULT_STREAM,
            );
        }

        let mut cumulative_sums =
            vec![SepticCurve::<BabyBear>::default(); trace.height()].to_device().unwrap();

        unsafe {
            tracegen::ffi::core_memory_local_generate_trace_round_2(
                trace_device.view_mut(),
                cumulative_sums.as_mut_ptr(),
                DEFAULT_STREAM,
            );
        }

        unsafe {
            tracegen::ffi::core_memory_local_generate_trace_round_3(
                trace_device.view_mut(),
                cumulative_sums.as_ptr(),
                nb_events as u32,
                DEFAULT_STREAM,
            );
        }

        let gpu_trace = trace_device.to_host();
        assert_eq!(trace, gpu_trace);
    }
}
