use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaError, memory::ToDevice},
    matrix::ColMajorMatrixDevice,
    tracegen,
};
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use sp1_core_machine::utils::next_power_of_two;
use sp1_recursion_core::{
    chips::{
        alu_base::{BaseAluChip, NUM_BASE_ALU_ENTRIES_PER_ROW, NUM_BASE_ALU_PREPROCESSED_COLS},
        alu_ext::{ExtAluChip, NUM_EXT_ALU_ENTRIES_PER_ROW},
        batch_fri::BatchFRIChip,
        fri_fold::FriFoldChip,
        poseidon2_skinny::{trace::OUTPUT_ROUND_IDX, Poseidon2SkinnyChip},
        poseidon2_wide::Poseidon2WideChip,
        public_values::PublicValuesChip,
        select::SelectChip,
    },
    runtime::{instruction as instr, Instruction, RecursionProgram},
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

        let nb_rows = instrs.len().div_ceil(NUM_BASE_ALU_ENTRIES_PER_ROW);
        let fixed_log2_rows = program.fixed_log2_rows(self);
        let padded_nb_rows = match fixed_log2_rows {
            Some(log2_rows) => 1 << log2_rows,
            None => next_power_of_two(nb_rows, None),
        };
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

        let nb_rows = instrs.len().div_ceil(NUM_EXT_ALU_ENTRIES_PER_ROW);
        let fixed_log2_rows = program.fixed_log2_rows(self);
        let padded_nb_rows = match fixed_log2_rows {
            Some(log2_rows) => 1 << log2_rows,
            None => next_power_of_two(nb_rows, None),
        };
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

impl DevicePreprocessedAir<BabyBear> for PublicValuesChip {
    fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let instrs = program
            .instructions
            .iter()
            .filter_map(|instruction| match instruction {
                Instruction::CommitPublicValues(x) => Some(**x),
                _ => None,
            })
            .collect::<Vec<_>>();
        let instrs = instrs.to_device_async(stream)?;

        let nb_rows = instrs.len().div_ceil(NUM_EXT_ALU_ENTRIES_PER_ROW);
        let fixed_log2_rows = program.fixed_log2_rows(self);
        let padded_nb_rows = match fixed_log2_rows {
            Some(log2_rows) => 1 << log2_rows,
            None => next_power_of_two(nb_rows, None),
        };
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <PublicValuesChip as MachineAir<BabyBear>>::preprocessed_width(self),
            padded_nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_public_values_generate_preprocessed_trace(
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

        let nb_rows = instrs.len();
        let fixed_log2_rows = program.fixed_log2_rows(self);
        let padded_nb_rows = match fixed_log2_rows {
            Some(log2_rows) => 1 << log2_rows,
            None => next_power_of_two(nb_rows, None),
        };
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

#[cfg(test)]
mod tests {
    use crate::tracegen::DeviceAir;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::Permutation;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use serial_test::serial;
    use sp1_recursion_core::{
        air::{Block, RecursionPublicValues, RECURSIVE_PROOF_NUM_PV_ELTS},
        chips::poseidon2_skinny::WIDTH,
        Address, BaseAluInstr, BaseAluIo, BaseAluOpcode, BatchFRIBaseVecIo, BatchFRIEvent,
        BatchFRIExtSingleIo, BatchFRIExtVecIo, CommitPublicValuesEvent, CommitPublicValuesInstr,
        ExecutionRecord, ExtAluInstr, ExtAluIo, ExtAluOpcode, FriFoldBaseIo, FriFoldEvent,
        FriFoldExtSingleIo, FriFoldExtVecIo, Poseidon2Event, SelectInstr, SelectIo,
    };
    use sp1_stark::{air::MachineAir, inner_perm};
    use std::{array, borrow::Borrow};
    use zkhash::ark_ff::UniformRand;

    use super::*;

    #[test]
    #[serial]
    fn test_base_alu() {
        type F = BabyBear;

        let chip = BaseAluChip;
        let program = RecursionProgram {
            instructions: vec![Instruction::BaseAlu(BaseAluInstr {
                opcode: BaseAluOpcode::AddF,
                mult: F::one(),
                addrs: BaseAluIo {
                    out: Address(F::zero()),
                    in1: Address(F::one()),
                    in2: Address(F::two()),
                },
            })],
            ..Default::default()
        };
        let trace = chip.generate_preprocessed_trace_host(&program).unwrap();

        let device_trace = chip
            .generate_preprocessed_trace_device(&program, &CudaStream::default())
            .unwrap()
            .unwrap();
        assert_eq!(trace, device_trace.to_host_naive());
    }

    #[test]
    #[serial]
    fn test_ext_alu() {
        type F = BabyBear;

        let chip = ExtAluChip;
        let program = RecursionProgram {
            instructions: vec![Instruction::ExtAlu(ExtAluInstr {
                opcode: ExtAluOpcode::AddE,
                mult: F::one(),
                addrs: ExtAluIo {
                    out: Address(F::zero()),
                    in1: Address(F::one()),
                    in2: Address(F::two()),
                },
            })],
            ..Default::default()
        };
        let trace = chip.generate_preprocessed_trace_host(&program).unwrap();

        let device_trace = chip
            .generate_preprocessed_trace_device(&program, &CudaStream::default())
            .unwrap()
            .unwrap();
        assert_eq!(trace, device_trace.to_host_naive());
    }

    #[test]
    #[serial]
    fn test_public_values() {
        type F = BabyBear;

        let chip = PublicValuesChip;
        let addr = 0u32;
        let public_values_a: [u32; RECURSIVE_PROOF_NUM_PV_ELTS] =
            array::from_fn(|i| i as u32 + addr);
        let public_values: &RecursionPublicValues<u32> = public_values_a.as_slice().borrow();
        let program = RecursionProgram {
            instructions: vec![instr::commit_public_values(public_values)],
            ..Default::default()
        };
        let trace = chip.generate_preprocessed_trace_host(&program).unwrap();

        let device_trace = chip
            .generate_preprocessed_trace_device(&program, &CudaStream::default())
            .unwrap()
            .unwrap();
        assert_eq!(trace, device_trace.to_host_naive());
    }

    #[test]
    #[serial]
    fn test_select() {
        type F = BabyBear;

        let chip = PublicValuesChip;
        let program = RecursionProgram {
            instructions: vec![
                Instruction::Select(SelectInstr {
                    addrs: SelectIo {
                        bit: Address(F::zero()),
                        out1: Address(F::one()),
                        out2: Address(F::from_canonical_u32(2)),
                        in1: Address(F::from_canonical_u32(3)),
                        in2: Address(F::from_canonical_u32(4)),
                    },
                    mult1: F::one(),
                    mult2: F::one(),
                }),
                Instruction::Select(SelectInstr {
                    addrs: SelectIo {
                        bit: Address(F::from_canonical_u32(5)),
                        out1: Address(F::from_canonical_u32(6)),
                        out2: Address(F::from_canonical_u32(7)),
                        in1: Address(F::from_canonical_u32(8)),
                        in2: Address(F::from_canonical_u32(9)),
                    },
                    mult1: F::one(),
                    mult2: F::one(),
                }),
            ],
            ..Default::default()
        };
        let trace = chip.generate_preprocessed_trace_host(&program).unwrap();

        let device_trace = chip
            .generate_preprocessed_trace_device(&program, &CudaStream::default())
            .unwrap()
            .unwrap();
        assert_eq!(trace, device_trace.to_host_naive());
    }
}
