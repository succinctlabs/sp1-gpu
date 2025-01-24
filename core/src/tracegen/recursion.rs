use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaError, memory::ToDevice},
    matrix::ColMajorMatrixDevice,
    tracegen,
};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use sp1_recursion_core::chips::{
    alu_base::BaseAluChip, alu_ext::ExtAluChip, batch_fri::BatchFRIChip, fri_fold::FriFoldChip,
    poseidon2_skinny::Poseidon2SkinnyChip, poseidon2_wide::Poseidon2WideChip, select::SelectChip,
};
use sp1_stark::air::MachineAir;

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
}

impl DeviceAir<BabyBear> for ExtAluChip {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let events = &input.ext_alu_events;
        let nb_rows = self.num_rows(input).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <ExtAluChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;
        let events = events.to_device_async(stream)?;

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
}

impl<const DEGREE: usize> DeviceAir<BabyBear> for FriFoldChip<DEGREE> {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let events = &input.fri_fold_events;
        let events = events.to_device_async(stream)?;

        let nb_rows = self.num_rows(input).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <FriFoldChip<DEGREE> as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_fri_fold_generate_trace(
                trace.view_mut(),
                events.as_ptr(),
                events.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }
}

impl DeviceAir<BabyBear> for SelectChip {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let events = &input.select_events;
        let events = events.to_device_async(stream)?;

        let nb_rows = self.num_rows(input).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <SelectChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_select_generate_trace(
                trace.view_mut(),
                events.as_ptr(),
                events.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }
}

impl<const DEGREE: usize> DeviceAir<BabyBear> for Poseidon2SkinnyChip<DEGREE> {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let events = &input.poseidon2_events;
        let events = events.to_device_async(stream)?;

        let nb_rows = self.num_rows(input).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <Poseidon2SkinnyChip<DEGREE> as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_poseidon2_skinny_generate_trace(
                trace.view_mut(),
                events.as_ptr(),
                events.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }
}

impl<const DEGREE: usize> DeviceAir<BabyBear> for Poseidon2WideChip<DEGREE> {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let events = &input.poseidon2_events;
        let events = events.to_device_async(stream)?;

        let nb_rows = self.num_rows(input).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <Poseidon2WideChip<DEGREE> as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_poseidon2_wide_generate_trace(
                trace.view_mut(),
                events.as_ptr(),
                events.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::tracegen::DeviceAir;
//     use serial_test::serial;
//     use sp1_recursion_core::{chips::test_fixtures, ExecutionRecord};
//     use sp1_stark::air::MachineAir;

//     use super::*;

//     #[test]
//     #[serial]
//     fn test_base_alu() {
//         let shard = test_fixtures::shard();
//         let trace = BaseAluChip.generate_trace(&shard, &mut ExecutionRecord::default());
//         let device_trace = BaseAluChip
//             .generate_trace_device(&shard, &mut ExecutionRecord::default(), &CudaStream::default())
//             .unwrap()
//             .unwrap();

//         assert_eq!(trace, device_trace.to_host_naive());
//     }

//     #[test]
//     #[serial]
//     fn test_ext_alu() {
//         let shard = test_fixtures::shard();
//         let trace = ExtAluChip.generate_trace(&shard, &mut ExecutionRecord::default());
//         let device_trace = ExtAluChip
//             .generate_trace_device(&shard, &mut ExecutionRecord::default(), &CudaStream::default())
//             .unwrap()
//             .unwrap();

//         assert_eq!(trace, device_trace.to_host_naive());
//     }

//     #[test]
//     #[serial]
//     fn test_batch_fri() {
//         let chip = BatchFRIChip::<2>;
//         let shard = test_fixtures::shard();
//         let trace = chip.generate_trace(&shard, &mut ExecutionRecord::default());

//         let device_trace = chip
//             .generate_trace_device(&shard, &mut ExecutionRecord::default(), &CudaStream::default())
//             .unwrap()
//             .unwrap();
//         assert_eq!(trace, device_trace.to_host_naive());
//     }

//     #[test]
//     #[serial]
//     fn test_fri_fold() {
//         let chip = FriFoldChip::<3>::default();
//         let shard = test_fixtures::shard();
//         let trace = chip.generate_trace(&shard, &mut ExecutionRecord::default());
//         let device_trace = chip
//             .generate_trace_device(&shard, &mut ExecutionRecord::default(), &CudaStream::default())
//             .unwrap()
//             .unwrap();

//         assert_eq!(trace, device_trace.to_host_naive());
//     }

//     #[test]
//     #[serial]
//     fn test_select() {
//         let shard = test_fixtures::shard();
//         let trace = SelectChip.generate_trace(&shard, &mut ExecutionRecord::default());
//         let device_trace = SelectChip
//             .generate_trace_device(&shard, &mut ExecutionRecord::default(), &CudaStream::default())
//             .unwrap()
//             .unwrap();

//         assert_eq!(trace, device_trace.to_host_naive());
//     }

//     #[test]
//     #[serial]
//     fn test_poseidon2_skinny() {
//         let chip = Poseidon2SkinnyChip::<9>::default();
//         let shard = test_fixtures::shard();
//         let trace = chip.generate_trace(&shard, &mut ExecutionRecord::default());
//         let device_trace = chip
//             .generate_trace_device(&shard, &mut ExecutionRecord::default(), &CudaStream::default())
//             .unwrap()
//             .unwrap();

//         assert_eq!(trace, device_trace.to_host_naive());
//     }

//     #[test]
//     #[serial]
//     fn test_poseidon2_wide_deg_3() {
//         let chip = Poseidon2WideChip::<3>;
//         let shard = test_fixtures::shard();
//         let trace = chip.generate_trace(&shard, &mut ExecutionRecord::default());
//         let device_trace = chip
//             .generate_trace_device(&shard, &mut ExecutionRecord::default(), &CudaStream::default())
//             .unwrap()
//             .unwrap();

//         assert_eq!(trace, device_trace.to_host_naive());
//     }

//     #[test]
//     #[serial]
//     fn test_poseidon2_wide_deg_9() {
//         let chip = Poseidon2WideChip::<9>;
//         let shard = test_fixtures::shard();
//         let trace = chip.generate_trace(&shard, &mut ExecutionRecord::default());
//         let device_trace = chip
//             .generate_trace_device(&shard, &mut ExecutionRecord::default(), &CudaStream::default())
//             .unwrap()
//             .unwrap();

//         assert_eq!(trace, device_trace.to_host_naive());
//     }
// }
