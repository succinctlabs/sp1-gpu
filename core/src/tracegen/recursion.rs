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
    exp_reverse_bits::ExpReverseBitsLenChip,
    fri_fold::FriFoldChip,
    poseidon2_skinny::{trace::OUTPUT_ROUND_IDX, Poseidon2SkinnyChip},
    poseidon2_wide::Poseidon2WideChip,
    public_values::PublicValuesChip,
    select::SelectChip,
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
        let nb_rows = events.len().div_ceil(NUM_BASE_ALU_ENTRIES_PER_ROW);
        let fixed_log2_rows = input.fixed_log2_rows(self);
        Some(match fixed_log2_rows {
            Some(log2_rows) => 1 << log2_rows,
            None => next_power_of_two(nb_rows, None),
        })
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
        let events = &input.ext_alu_events;
        let nb_rows = events.len().div_ceil(NUM_EXT_ALU_ENTRIES_PER_ROW);
        let fixed_log2_rows = input.fixed_log2_rows(self);
        Some(match fixed_log2_rows {
            Some(log2_rows) => 1 << log2_rows,
            None => next_power_of_two(nb_rows, None),
        })
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
        let events = &input.batch_fri_events;
        Some(next_power_of_two(events.len().div_ceil(1), input.fixed_log2_rows(self)))
    }
}

impl<const DEGREE: usize> DeviceAir<BabyBear> for ExpReverseBitsLenChip<DEGREE> {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let events = &input.exp_reverse_bits_len_events;
        let events = events.to_device_async(stream)?;

        let nb_rows = self.num_rows(input).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <ExpReverseBitsLenChip<DEGREE> as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_exp_reverse_bits_generate_trace(
                trace.view_mut(),
                events.as_ptr(),
                events.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.exp_reverse_bits_len_events;
        Some(next_power_of_two(
            events.iter().map(|e| e.len).sum::<usize>(),
            input.fixed_log2_rows(self),
        ))
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

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.fri_fold_events;
        Some(next_power_of_two(events.len().div_ceil(1), input.fixed_log2_rows(self)))
    }
}

impl DeviceAir<BabyBear> for PublicValuesChip {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        let events = &input.commit_pv_hash_events;
        let events = events.to_device_async(stream)?;

        let nb_rows = self.num_rows(input).unwrap();
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <PublicValuesChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        unsafe {
            trace.set_max_width();
            tracegen::ffi::recursion_public_values_generate_trace(
                trace.view_mut(),
                events.as_ptr(),
                events.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.commit_pv_hash_events;
        Some(next_power_of_two(events.len() * 8, input.fixed_log2_rows(self)))
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

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.select_events;
        Some(next_power_of_two(events.len().div_ceil(1), input.fixed_log2_rows(self)))
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

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.poseidon2_events;
        Some(next_power_of_two(events.len() * (OUTPUT_ROUND_IDX + 1), input.fixed_log2_rows(self)))
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

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.poseidon2_events;
        match input.fixed_log2_rows(self) {
            Some(log2_rows) => Some(1 << log2_rows),
            None => Some(next_power_of_two(events.len(), None)),
        }
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
        BaseAluIo, BatchFRIBaseVecIo, BatchFRIEvent, BatchFRIExtSingleIo, BatchFRIExtVecIo,
        CommitPublicValuesEvent, ExecutionRecord, ExpReverseBitsEvent, ExtAluIo, FriFoldBaseIo,
        FriFoldEvent, FriFoldExtSingleIo, FriFoldExtVecIo, Poseidon2Event, SelectIo,
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
        let shard = ExecutionRecord {
            base_alu_events: vec![
                BaseAluIo { out: F::one(), in1: F::one(), in2: F::one() },
                BaseAluIo { out: F::one(), in1: F::zero(), in2: F::one() },
            ],
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
    #[serial]
    fn test_ext_alu() {
        type F = BabyBear;

        let chip = ExtAluChip;
        let shard = ExecutionRecord {
            ext_alu_events: vec![
                ExtAluIo { out: F::one().into(), in1: F::one().into(), in2: F::one().into() },
                ExtAluIo { out: F::one().into(), in1: F::zero().into(), in2: F::one().into() },
            ],
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
    #[serial]
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

    #[test]
    #[serial]
    fn test_exp_reverse_bits() {
        type F = BabyBear;

        let chip = ExpReverseBitsLenChip::<3>;
        let shard = ExecutionRecord {
            exp_reverse_bits_len_events: vec![ExpReverseBitsEvent {
                base: F::two(),
                exp: to_fixed_array(vec![F::zero(), F::one(), F::one()]),
                len: 3,
                result: F::two().exp_u64(0b110),
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

    fn to_fixed_array(vec: Vec<BabyBear>) -> [BabyBear; 32] {
        let mut arr = [BabyBear::zero(); 32];
        for (i, val) in vec.into_iter().take(32).enumerate() {
            arr[i] = val;
        }
        arr
    }

    #[test]
    #[serial]
    fn test_fri_fold() {
        type F = BabyBear;

        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut rng2 = StdRng::seed_from_u64(0xDEADBEEF);
        let mut random_felt = move || -> F { F::from_canonical_u32(rng.gen_range(0..1 << 16)) };
        let mut random_block = move || Block::from([random_felt(); 4]);

        let chip = FriFoldChip::<3>::default();
        let shard = ExecutionRecord {
            fri_fold_events: (0..17)
                .map(|_| FriFoldEvent {
                    base_single: FriFoldBaseIo {
                        x: F::from_canonical_u32(rng2.gen_range(0..1 << 16)),
                    },
                    ext_single: FriFoldExtSingleIo { z: random_block(), alpha: random_block() },
                    ext_vec: FriFoldExtVecIo {
                        mat_opening: random_block(),
                        ps_at_z: random_block(),
                        alpha_pow_input: random_block(),
                        ro_input: random_block(),
                        alpha_pow_output: random_block(),
                        ro_output: random_block(),
                    },
                })
                .collect(),
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
    #[serial]
    fn test_public_values() {
        type F = BabyBear;

        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let random_felts: [F; RECURSIVE_PROOF_NUM_PV_ELTS] =
            array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..1 << 16)));
        let random_public_values: &RecursionPublicValues<F> = random_felts.as_slice().borrow();

        let chip = PublicValuesChip;
        let shard = ExecutionRecord {
            commit_pv_hash_events: vec![CommitPublicValuesEvent {
                public_values: *random_public_values,
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
    #[serial]
    fn test_select() {
        type F = BabyBear;

        let chip = SelectChip;
        let shard = ExecutionRecord {
            select_events: vec![
                SelectIo {
                    bit: F::one(),
                    out1: F::from_canonical_u32(5),
                    out2: F::from_canonical_u32(3),
                    in1: F::from_canonical_u32(3),
                    in2: F::from_canonical_u32(5),
                },
                SelectIo {
                    bit: F::zero(),
                    out1: F::from_canonical_u32(5),
                    out2: F::from_canonical_u32(3),
                    in1: F::from_canonical_u32(5),
                    in2: F::from_canonical_u32(3),
                },
            ],
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
    #[serial]
    fn test_poseidon2_skinny() {
        type F = BabyBear;

        let chip = Poseidon2SkinnyChip::<9>::default();
        let input_0 = [F::one(); WIDTH];
        let permuter = inner_perm();
        let output_0 = permuter.permute(input_0);
        let mut rng = rand::thread_rng();

        let input_1 = [F::rand(&mut rng); WIDTH];
        let output_1 = permuter.permute(input_1);
        let shard = ExecutionRecord {
            poseidon2_events: vec![
                Poseidon2Event { input: input_0, output: output_0 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
                Poseidon2Event { input: input_1, output: output_1 },
            ],
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
    #[serial]
    fn test_poseidon2_wide_deg_3() {
        type F = BabyBear;

        let chip = Poseidon2WideChip::<3>;
        let input_0 = [F::one(); WIDTH];
        let permuter = inner_perm();
        let output_0 = permuter.permute(input_0);
        let mut rng = rand::thread_rng();

        let input_1 = [F::rand(&mut rng); WIDTH];
        let output_1 = permuter.permute(input_1);
        let shard = ExecutionRecord {
            poseidon2_events: vec![
                Poseidon2Event { input: input_0, output: output_0 },
                Poseidon2Event { input: input_1, output: output_1 },
            ],
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
    #[serial]
    fn test_poseidon2_wide_deg_9() {
        type F = BabyBear;

        let chip = Poseidon2WideChip::<9>;
        let input_0 = [F::one(); WIDTH];
        let permuter = inner_perm();
        let output_0 = permuter.permute(input_0);
        let mut rng = rand::thread_rng();

        let input_1 = [F::rand(&mut rng); WIDTH];
        let output_1 = permuter.permute(input_1);
        let shard = ExecutionRecord {
            poseidon2_events: vec![
                Poseidon2Event { input: input_0, output: output_0 },
                Poseidon2Event { input: input_1, output: output_1 },
            ],
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
