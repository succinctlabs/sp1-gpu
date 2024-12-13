use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaError, memory::ToDevice},
    matrix::ColMajorMatrixDevice,
    tracegen,
};
use p3_baby_bear::BabyBear;
use sp1_recursion_core::{
    air::Block,
    chips::{
        alu_base::BaseAluChip,
        alu_ext::ExtAluChip,
        mem::{MemoryAccessCols, MemoryAccessColsChips, MemoryConstChip, MemoryVarChip},
        poseidon2_skinny::{Poseidon2SkinnyChip, NUM_EXTERNAL_ROUNDS},
        poseidon2_wide::Poseidon2WideChip,
        select::SelectChip,
    },
    instruction::{HintAddCurveInstr, HintBitsInstr, HintExt2FeltsInstr, HintInstr},
    runtime::Instruction,
    Address, MemAccessKind, MemInstr,
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
            .inner
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
            .inner
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
            .inner
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
            .inner
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
            .inner
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

impl DevicePreprocessedAir<BabyBear> for MemoryVarChip<BabyBear> {
    fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let (addresses, multipliers): (Vec<Address<BabyBear>>, Vec<BabyBear>) = program
            .inner
            .iter()
            .flat_map(|instruction| match instruction {
                Instruction::Hint(HintInstr { output_addrs_mults })
                | Instruction::HintBits(HintBitsInstr { output_addrs_mults, input_addr: _ }) => {
                    output_addrs_mults.iter().collect()
                }
                Instruction::HintExt2Felts(HintExt2FeltsInstr {
                    output_addrs_mults,
                    input_addr: _,
                }) => output_addrs_mults.iter().collect(),
                Instruction::HintAddCurve(instr) => {
                    let HintAddCurveInstr { output_x_addrs_mults, output_y_addrs_mults, .. } =
                        instr.as_ref();
                    output_x_addrs_mults.iter().chain(output_y_addrs_mults.iter()).collect()
                }
                _ => vec![],
            })
            .map(|(addr, mult)| (addr, mult))
            .unzip();

        let instrs_len = addresses.len();
        let padded_nb_rows = self.preprocessed_num_rows(program, instrs_len).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <MemoryVarChip<BabyBear> as MachineAir<BabyBear>>::preprocessed_width(self),
            padded_nb_rows,
            stream,
        )?;

        let addresses = addresses.to_device_async(stream)?;
        let multipliers = multipliers.to_device_async(stream)?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_mem_variable_generate_preprocessed_trace(
                trace.view_mut(),
                addresses.as_ptr(),
                multipliers.as_ptr(),
                instrs_len as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }
}

impl DevicePreprocessedAir<BabyBear> for MemoryConstChip<BabyBear> {
    fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let (blocks, access_cols): (Vec<Block<BabyBear>>, Vec<MemoryAccessColsChips<BabyBear>>) =
            program
                .inner
                .iter()
                .filter_map(|instruction| match instruction {
                    Instruction::Mem(MemInstr { addrs, vals, mult, kind }) => {
                        let mult = mult.to_owned();
                        let mult = match kind {
                            MemAccessKind::Read => -mult,
                            MemAccessKind::Write => mult,
                        };

                        Some((vals.inner, MemoryAccessCols { addr: addrs.inner, mult }))
                    }
                    _ => None,
                })
                .unzip();

        let instrs_len = blocks.len();
        let padded_nb_rows = self.preprocessed_num_rows(program, instrs_len).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <MemoryConstChip<BabyBear> as MachineAir<BabyBear>>::preprocessed_width(self),
            padded_nb_rows,
            stream,
        )?;

        let blocks = blocks.to_device_async(stream)?;
        let access_cols = access_cols.to_device_async(stream)?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_mem_const_generate_preprocessed_trace(
                trace.view_mut(),
                blocks.as_ptr(),
                access_cols.as_ptr(),
                instrs_len as u32,
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

    #[test]
    #[serial]
    fn test_mem_variable() {
        let chip = MemoryVarChip::<BabyBear>::default();
        let program = test_fixtures::program();
        let trace = chip.generate_preprocessed_trace_host(&program).unwrap();
        let device_trace = chip
            .generate_preprocessed_trace_device(&program, &CudaStream::default())
            .unwrap()
            .unwrap();

        assert_eq!(trace, device_trace.to_host_naive());
    }
}
