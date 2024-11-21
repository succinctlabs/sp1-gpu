use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaError, memory::ToDevice},
    matrix::ColMajorMatrixDevice,
    tracegen,
};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use sp1_core_machine::utils::next_power_of_two;
use sp1_recursion_core::chips::{
    alu_base::{BaseAluChip, NUM_BASE_ALU_ENTRIES_PER_ROW},
    alu_ext::{ExtAluChip, NUM_EXT_ALU_ENTRIES_PER_ROW},
    batch_fri::BatchFRIChip,
};

use super::DeviceAir;

impl DeviceAir<BabyBear> for BaseAluChip {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let events = &input.base_alu_events;
        let events = events.to_device_async(stream)?;

        let nb_rows = self.num_rows(input).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <BaseAluChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

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

impl DeviceAir<BabyBear> for ExtAluChip {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let events = &input.ext_alu_events;
        let events = events.to_device_async(stream)?;

        let nb_rows = self.num_rows(input).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <ExtAluChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_ext_alu_generate_trace(
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
            events.len().div_ceil(NUM_EXT_ALU_ENTRIES_PER_ROW),
            input.fixed_log2_rows(self),
        ))
    }
}

impl<const DEGREE: usize> DeviceAir<BabyBear> for BatchFRIChip<DEGREE> {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let events = &input.batch_fri_events;
        let events = events.to_device_async(stream)?;

        let nb_rows = self.num_rows(input).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <BatchFRIChip<DEGREE> as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_batch_fri_generate_trace(
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
        Some(next_power_of_two(events.len().div_ceil(1), input.fixed_log2_rows(self)))
    }
}

#[cfg(test)]
mod tests {
    use crate::tracegen::DeviceAir;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;
    use sp1_recursion_core::{
        air::Block, BaseAluIo, BatchFRIBaseVecIo, BatchFRIEvent, BatchFRIExtSingleIo,
        BatchFRIExtVecIo, ExecutionRecord, ExtAluIo,
    };
    use sp1_stark::air::MachineAir;

    use super::*;

    #[test]
    fn test_base_alu() {
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

    #[test]
    fn test_ext_alu() {
        type F = BabyBear;

        let chip = ExtAluChip;
        let shard = ExecutionRecord {
            ext_alu_events: vec![ExtAluIo {
                out: F::one().into(),
                in1: F::one().into(),
                in2: F::one().into(),
            }],
            ..Default::default()
        };
        let trace: RowMajorMatrix<F> = chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let device_trace = chip
            .generate_trace_device(&shard, &mut ExecutionRecord::default(), &CudaStream::default())
            .unwrap()
            .unwrap();
        assert_eq!(trace, device_trace.to_host_naive());
    }

    #[test]
    fn test_batch_fri() {
        type F = BabyBear;

        let chip = BatchFRIChip::<2>;
        let shard = ExecutionRecord {
            batch_fri_events: vec![BatchFRIEvent {
                ext_single: BatchFRIExtSingleIo { acc: Block::default() },
                ext_vec: BatchFRIExtVecIo { alpha_pow: Block::default(), p_at_z: Block::default() },
                base_vec: BatchFRIBaseVecIo { p_at_x: F::one() },
            }],
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
