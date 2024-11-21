use crate::cuda_runtime::stream::CudaStreamHandle;
use crate::matrix::MatrixViewMutDevice;
use p3_baby_bear::BabyBear;
use sp1_core_executor::events::AluEvent;

/// cbindgen:ignore
#[allow(unused_attributes)]
#[link_name = "moongate::add_sub"]
extern "C" {
    pub fn generate_trace(
        trace: MatrixViewMutDevice<BabyBear>,
        events: *const AluEvent,
        nb_events: u32,
        stream: CudaStreamHandle,
    );
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

    use super::generate_trace;

    #[test]
    fn test_add_sub_generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.add_events = vec![AluEvent::new(0, 0, Opcode::ADD, 14, 8, 6)].repeat(100);

        let chip = AddSubChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let mut trace_device =
            RowMajorMatrixDevice::new(trace.values.to_device().unwrap(), trace.width())
                .to_column_major();

        let events = shard.add_events.to_device().unwrap().as_ptr();
        unsafe {
            generate_trace(
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
