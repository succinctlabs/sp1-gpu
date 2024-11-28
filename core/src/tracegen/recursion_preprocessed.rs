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
    runtime::{Instruction, RecursionProgram},
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
        BatchFRIExtSingleIo, BatchFRIExtVecIo, CommitPublicValuesEvent, ExecutionRecord, ExtAluIo,
        FriFoldBaseIo, FriFoldEvent, FriFoldExtSingleIo, FriFoldExtVecIo, Poseidon2Event, SelectIo,
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
}
