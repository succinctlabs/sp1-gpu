use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaError, memory::ToDevice},
    matrix::ColMajorMatrixDevice,
    tracegen,
};
use p3_baby_bear::BabyBear;
use sp1_recursion_core::{
    chips::{
        alu_base::BaseAluChip,
        alu_ext::ExtAluChip,
        poseidon2_skinny::{Poseidon2SkinnyChip, NUM_EXTERNAL_ROUNDS},
        poseidon2_wide::Poseidon2WideChip,
        select::SelectChip,
    },
    runtime::Instruction,
};
use sp1_stark::air::MachineAir;

use super::DevicePreprocessedAir;

impl DevicePreprocessedAir<BabyBear> for BaseAluChip {
    fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let instrs = program
            .instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::BaseAlu(x) => Some(*x),
                _ => None,
            })
            .collect::<Vec<_>>();
        let instrs = instrs.to_device_async(stream)?;

        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <BaseAluChip as MachineAir<BabyBear>>::preprocessed_width(self),
            padded_nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_base_alu_generate_preprocessed_trace(
                trace.view_mut(),
                instrs.as_ptr(),
                instrs.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }
}

impl DevicePreprocessedAir<BabyBear> for ExtAluChip {
    fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let instrs = program
            .instructions
            .iter() // Faster than using `rayon` for some reason. Maybe vectorization?
            .filter_map(|instruction| match instruction {
                Instruction::ExtAlu(x) => Some(*x),
                _ => None,
            })
            .collect::<Vec<_>>();
        let instrs = instrs.to_device_async(stream)?;

        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <ExtAluChip as MachineAir<BabyBear>>::preprocessed_width(self),
            padded_nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_ext_alu_generate_preprocessed_trace(
                trace.view_mut(),
                instrs.as_ptr(),
                instrs.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }
}

impl DevicePreprocessedAir<BabyBear> for SelectChip {
    fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let instrs = program
            .instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::Select(x) => Some(*x),
                _ => None,
            })
            .collect::<Vec<_>>();
        let instrs = instrs.to_device_async(stream)?;

        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <SelectChip as MachineAir<BabyBear>>::preprocessed_width(self),
            padded_nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_select_generate_preprocessed_trace(
                trace.view_mut(),
                instrs.as_ptr(),
                instrs.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }
}

impl<const DEGREE: usize> DevicePreprocessedAir<BabyBear> for Poseidon2SkinnyChip<DEGREE> {
    fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let instrs = program
            .instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::Poseidon2(instr) => Some(**instr),
                _ => None,
            })
            .collect::<Vec<_>>();
        let instrs = instrs.to_device_async(stream)?;

        let padded_nb_rows =
            self.preprocessed_num_rows(program, instrs.len() * (NUM_EXTERNAL_ROUNDS + 3)).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <Poseidon2SkinnyChip<DEGREE> as MachineAir<BabyBear>>::preprocessed_width(self),
            padded_nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_poseidon2_skinny_generate_preprocessed_trace(
                trace.view_mut(),
                instrs.as_ptr(),
                instrs.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }
}

impl<const DEGREE: usize> DevicePreprocessedAir<BabyBear> for Poseidon2WideChip<DEGREE> {
    fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let instrs = program
            .instructions
            .iter() // Faster than using `rayon` for some reason. Maybe vectorization?
            .filter_map(|instruction| match instruction {
                Instruction::Poseidon2(instr) => Some(**instr),
                _ => None,
            })
            .collect::<Vec<_>>();
        let instrs = instrs.to_device_async(stream)?;

        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <Poseidon2WideChip<DEGREE> as MachineAir<BabyBear>>::preprocessed_width(self),
            padded_nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_poseidon2_wide_generate_preprocessed_trace(
                trace.view_mut(),
                instrs.as_ptr(),
                instrs.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;
    use sp1_recursion_core::chips::test_fixtures;

    use super::*;

    #[test]
    #[serial]
    fn test_base_alu() {
        let program = test_fixtures::program();
        let trace = BaseAluChip.generate_preprocessed_trace_host(&program).unwrap();
        let device_trace = BaseAluChip
            .generate_preprocessed_trace_device(&program, &CudaStream::default())
            .unwrap()
            .unwrap();

        assert_eq!(trace, device_trace.to_host_naive());
    }

    #[test]
    #[serial]
    fn test_ext_alu() {
        let program = test_fixtures::program();
        let trace = ExtAluChip.generate_preprocessed_trace_host(&program).unwrap();
        let device_trace = ExtAluChip
            .generate_preprocessed_trace_device(&program, &CudaStream::default())
            .unwrap()
            .unwrap();

        assert_eq!(trace, device_trace.to_host_naive());
    }

    #[test]
    #[serial]
    fn test_select() {
        let program = test_fixtures::program();
        let trace = SelectChip.generate_preprocessed_trace_host(&program).unwrap();
        let device_trace = SelectChip
            .generate_preprocessed_trace_device(&program, &CudaStream::default())
            .unwrap()
            .unwrap();

        assert_eq!(trace, device_trace.to_host_naive());
    }

    #[test]
    #[serial]
    fn test_poseidon2_skinny() {
        let chip = Poseidon2SkinnyChip::<9>::default();
        let program = test_fixtures::program();
        let trace = chip.generate_preprocessed_trace_host(&program).unwrap();
        let device_trace = chip
            .generate_preprocessed_trace_device(&program, &CudaStream::default())
            .unwrap()
            .unwrap();

        assert_eq!(trace, device_trace.to_host_naive());
    }

    #[test]
    #[serial]
    fn test_poseidon2_wide_deg_3() {
        let chip = Poseidon2WideChip::<3>;
        let program = test_fixtures::program();
        let trace = chip.generate_preprocessed_trace_host(&program).unwrap();
        let device_trace = chip
            .generate_preprocessed_trace_device(&program, &CudaStream::default())
            .unwrap()
            .unwrap();

        assert_eq!(trace, device_trace.to_host_naive());
    }

    #[test]
    #[serial]
    fn test_poseidon2_wide_deg_9() {
        let chip = Poseidon2WideChip::<9>;
        let program = test_fixtures::program();
        let trace = chip.generate_preprocessed_trace_host(&program).unwrap();
        let device_trace = chip
            .generate_preprocessed_trace_device(&program, &CudaStream::default())
            .unwrap()
            .unwrap();

        assert_eq!(trace, device_trace.to_host_naive());
    }
}
