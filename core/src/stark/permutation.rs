use std::{marker::PhantomData, ops::Mul};

use p3_air::PairCol;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractExtensionField, Field};
use sp1_stark::{
    air::{InteractionScope, MachineAir},
    septic_curve::SepticCurve,
    septic_digest::SepticDigest,
    septic_extension::SepticExtension,
    Chip, Interaction,
};

use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaError, memory::ToDevice, slice::DeviceSlice, DeviceBuffer},
    matrix::{ColMajorMatrixDevice, MatrixViewDevice, MatrixViewMutDevice},
};

use super::ffi;

#[derive(Clone, Copy, Debug)]
pub struct PermutationTraceGenerator<F, EF, A>(PhantomData<(F, EF, A)>);

impl<F, EF, A> Default for PermutationTraceGenerator<F, EF, A> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<A> PermutationTraceGenerator<BabyBear, BinomialExtensionField<BabyBear, 4>, A>
where
    A: MachineAir<BabyBear>,
{
    pub fn generate_flattened_permutation_trace(
        &self,
        chip: &Chip<BabyBear, A>,
        preprocessed_trace: Option<&ColMajorMatrixDevice<BabyBear>>,
        main_trace: &ColMajorMatrixDevice<BabyBear>,
        random_elements: &[BinomialExtensionField<BabyBear, 4>],
    ) -> Result<
        (
            ColMajorMatrixDevice<BabyBear>,
            (BinomialExtensionField<BabyBear, 4>, SepticDigest<BabyBear>),
        ),
        CudaError,
    > {
        let stream = main_trace.stream();
        const D: usize = 4;

        let batch_size = chip.logup_batch_size();

        let num_interactions = chip.sends().len() + chip.receives().len();

        let device_interactions =
            HostInteractions::new(chip.sends(), chip.receives()).to_device_async(stream).unwrap();

        let perm_width = chip.permutation_width();
        let height = main_trace.height;
        let mut perm_buffer =
            DeviceBuffer::<BabyBear>::with_capacity_in(perm_width * height * D, stream).unwrap();
        unsafe {
            perm_buffer.set_max_len();
        }
        let mut permutation_trace = ColMajorMatrixDevice::new(perm_buffer, height);

        let alpha = random_elements[0];
        let beta = random_elements[1];

        let num_threads_per_block = 256;
        let num_blocks = height.div_ceil(num_threads_per_block);

        device_interactions.generate_flattened_permutation_trace(
            permutation_trace.view_mut(),
            preprocessed_trace.map(|mat| mat.view()).unwrap_or(MatrixViewDevice::null(false)),
            main_trace.view(),
            alpha,
            beta,
            batch_size,
            num_blocks,
            num_threads_per_block,
            stream,
        )?;

        // Retrieve the cumulative sums.
        let row_idx = permutation_trace.height() - 1;
        let mut global_cumulative_sum = SepticDigest::<BabyBear>::zero();
        let mut local_cumulative_sum = Default::default();

        if num_interactions != 0 {
            let cumulative_sum = BinomialExtensionField::<BabyBear, 4>::from_base_fn(|i| {
                let index =
                    (permutation_trace.width() - D + i) * permutation_trace.height() + row_idx;
                let val = permutation_trace.values[index..index + 1]
                    .as_host_vec(permutation_trace.stream());
                val[0]
            });

            local_cumulative_sum = cumulative_sum;
        }

        if chip.commit_scope() == InteractionScope::Global {
            let x = SepticExtension::<BabyBear>::from_base_fn(|i| {
                let index =
                    (main_trace.width() - 14 + i) * main_trace.height() + main_trace.height() - 1;
                let val = main_trace.values[index..index + 1].as_host_vec(main_trace.stream());
                val[0]
            });

            let y = SepticExtension::<BabyBear>::from_base_fn(|i| {
                let index =
                    (main_trace.width() - 7 + i) * main_trace.height() + main_trace.height() - 1;
                let val = main_trace.values[index..index + 1].as_host_vec(main_trace.stream());
                val[0]
            });

            global_cumulative_sum = SepticDigest(SepticCurve { x, y });
        }

        Ok((permutation_trace, (local_cumulative_sum, global_cumulative_sum)))
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PairColDevice<F> {
    column_idx: usize,
    is_preprocessed: bool,
    weight: F,
}

#[derive(Debug)]
#[repr(C)]
pub struct HostInteractions<F: Field> {
    pub values_ptr: Vec<usize>,
    pub multiplicities_ptr: Vec<usize>,
    pub values_col_weights_ptr: Vec<usize>,

    pub values_col_weights: Vec<PairColDevice<F>>,
    pub values_constants: Vec<F>,

    pub mult_col_weights: Vec<PairColDevice<F>>,
    pub mult_constants: Vec<F>,

    pub arg_indices: Vec<F>,
    pub is_send: Vec<bool>,

    pub num_interactions: usize,
}

#[derive(Debug)]
#[repr(C)]
pub struct DeviceInteractions<F: Field> {
    pub values_ptr: DeviceBuffer<usize>,
    pub multiplicities_ptr: DeviceBuffer<usize>,
    pub values_col_weights_ptr: DeviceBuffer<usize>,

    pub values_col_weights: DeviceBuffer<PairColDevice<F>>,
    pub values_constants: DeviceBuffer<F>,

    pub mult_col_weights: DeviceBuffer<PairColDevice<F>>,
    pub mult_constants: DeviceBuffer<F>,

    pub arg_indices: DeviceBuffer<F>,
    pub is_send: DeviceBuffer<bool>,

    pub num_interactions: usize,
}

#[derive(Debug)]
#[repr(C)]
pub struct DeviceInteractionsView<'a, F: Field> {
    pub values_ptr: *const usize,
    pub multiplicities_ptr: *const usize,
    pub values_col_weights_ptr: *const usize,

    pub values_col_weights: *const PairColDevice<F>,
    pub values_constants: *const F,

    pub mult_col_weights: *const PairColDevice<F>,
    pub mult_constants: *const F,

    pub arg_indices: *const F,
    pub is_send: *const bool,

    pub num_interactions: usize,

    _marker: PhantomData<&'a F>,
}

impl<F: Field> PairColDevice<F> {
    pub fn get(&self, preprocessed: &[F], main: &[F]) -> F {
        if self.is_preprocessed {
            preprocessed[self.column_idx] * self.weight
        } else {
            main[self.column_idx] * self.weight
        }
    }
}

impl<F: Field> HostInteractions<F> {
    pub fn new(sends: &[Interaction<F>], receives: &[Interaction<F>]) -> Self {
        let mut values_ptr = vec![];
        let mut values_col_weights_ptr = vec![];
        let mut multiplicities_ptr = vec![];
        let mut arg_indices = vec![];
        let mut is_send = vec![];
        let mut mult_col_weights = vec![];
        let mut mult_constants = vec![];
        let mut values_col_weights = vec![];
        let mut values_constants = vec![];

        let num_interactions = sends.len() + receives.len();

        let mut curr_values_ptr = 0;
        let mut curr_values_col_weight_ptr = 0;
        let mut curr_mult_ptr = 0;

        // Put all of the interactions (for both send/receives) into a single list.
        // The ordering of the interactions is important to match with the CPU prover's ordering.
        // It should local sends, local receives.
        let interactions = {
            let sends = sends.iter().map(move |i| (i, true));
            let receives = receives.iter().map(move |i| (i, false));
            sends.chain(receives)
        };

        for (interaction, is_send_flag) in interactions {
            // Register the values
            values_ptr.push(curr_values_ptr);
            for value in interaction.values.iter() {
                values_col_weights_ptr.push(curr_values_col_weight_ptr);
                for (col, weight) in value.column_weights.iter() {
                    let col = PairColDevice::<F>::from(*col) * *weight;
                    values_col_weights.push(col);
                    curr_values_col_weight_ptr += 1;
                }
                values_constants.push(value.constant);
                curr_values_ptr += 1;
            }

            // Register the multiplicity values
            multiplicities_ptr.push(curr_mult_ptr);
            for (col, weight) in interaction.multiplicity.column_weights.iter() {
                let col = PairColDevice::<F>::from(*col) * *weight;
                mult_col_weights.push(col);
                curr_mult_ptr += 1;
            }
            mult_constants.push(interaction.multiplicity.constant);

            arg_indices.push(F::from_canonical_usize(interaction.argument_index()));

            is_send.push(is_send_flag);
        }

        values_col_weights_ptr.push(curr_values_col_weight_ptr);
        values_ptr.push(curr_values_ptr);
        multiplicities_ptr.push(curr_mult_ptr);

        Self {
            values_ptr,
            values_col_weights_ptr,
            multiplicities_ptr,
            values_col_weights,
            values_constants,
            mult_col_weights,
            mult_constants,
            arg_indices,
            is_send,
            num_interactions,
        }
    }
}

impl<F: Field> DeviceInteractions<F> {
    pub fn view(&self) -> DeviceInteractionsView<'_, F> {
        DeviceInteractionsView {
            values_ptr: self.values_ptr.as_ptr(),
            multiplicities_ptr: self.multiplicities_ptr.as_ptr(),
            values_col_weights_ptr: self.values_col_weights_ptr.as_ptr(),

            values_col_weights: self.values_col_weights.as_ptr(),
            values_constants: self.values_constants.as_ptr(),

            mult_col_weights: self.mult_col_weights.as_ptr(),
            mult_constants: self.mult_constants.as_ptr(),

            arg_indices: self.arg_indices.as_ptr(),
            is_send: self.is_send.as_ptr(),

            num_interactions: self.num_interactions,
            _marker: PhantomData,
        }
    }
}

impl DeviceInteractions<BabyBear> {
    pub fn populate_permutation_rows_flattened(
        &self,
        permutation: MatrixViewMutDevice<BabyBear>,
        preprocessed: MatrixViewDevice<BabyBear>,
        main: MatrixViewDevice<BabyBear>,
        alpha: BinomialExtensionField<BabyBear, 4>,
        beta: BinomialExtensionField<BabyBear, 4>,
        batch_size: usize,
        num_blocks: usize,
        num_threads_per_block: usize,
        stream: &CudaStream,
    ) {
        self.view().populate_permutation_rows_flattened(
            permutation,
            preprocessed,
            main,
            alpha,
            beta,
            batch_size,
            num_blocks,
            num_threads_per_block,
            stream,
        );
    }

    pub fn generate_flattened_permutation_trace(
        &self,
        permutation: MatrixViewMutDevice<BabyBear>,
        preprocessed: MatrixViewDevice<BabyBear>,
        main: MatrixViewDevice<BabyBear>,
        alpha: BinomialExtensionField<BabyBear, 4>,
        beta: BinomialExtensionField<BabyBear, 4>,
        batch_size: usize,
        num_blocks: usize,
        num_threads_per_block: usize,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        const D: usize = 4;
        // Populate the permutation rows.
        self.populate_permutation_rows_flattened(
            permutation,
            preprocessed,
            main,
            alpha,
            beta,
            batch_size,
            num_blocks,
            num_threads_per_block,
            stream,
        );

        // Collect the cumulative sums using a scan in place.
        // TODO: optimize with a single kernel call instead of scan for each column of the batch.
        let height = permutation.height;
        if permutation.width != 0 {
            let col = permutation.width - D;
            unsafe {
                for j in 0..4 {
                    let last_col_ptr = permutation.values.add((col + j) * height);
                    let cumulative_column = DeviceSlice::from_raw_parts_mut(last_col_ptr, height);
                    cumulative_column.scan_inplace(stream)?;
                }
            }
        }

        Ok(())
    }
}

impl<'a> DeviceInteractionsView<'a, BabyBear> {
    pub fn populate_permutation_rows_flattened(
        self,
        permutation: MatrixViewMutDevice<BabyBear>,
        preprocessed: MatrixViewDevice<BabyBear>,
        main: MatrixViewDevice<BabyBear>,
        alpha: BinomialExtensionField<BabyBear, 4>,
        beta: BinomialExtensionField<BabyBear, 4>,
        batch_size: usize,
        num_blocks: usize,
        num_threads_per_block: usize,
        stream: &CudaStream,
    ) {
        unsafe {
            ffi::populate_permutation_rows_flattened(
                self,
                permutation,
                preprocessed,
                main,
                alpha,
                beta,
                batch_size,
                num_blocks,
                num_threads_per_block,
                stream.handle(),
            );
        }
    }
}

impl<F: Field> From<PairCol> for PairColDevice<F> {
    fn from(value: PairCol) -> Self {
        match value {
            PairCol::Preprocessed(column_idx) => {
                Self { column_idx, is_preprocessed: true, weight: F::one() }
            }
            PairCol::Main(column_idx) => {
                Self { column_idx, is_preprocessed: false, weight: F::one() }
            }
        }
    }
}

impl<F: Field> Mul<F> for PairColDevice<F> {
    type Output = PairColDevice<F>;

    fn mul(self, rhs: F) -> Self::Output {
        PairColDevice {
            column_idx: self.column_idx,
            is_preprocessed: self.is_preprocessed,
            weight: self.weight * rhs,
        }
    }
}

impl<F: Field> ToDevice for HostInteractions<F> {
    type DeviceType = DeviceInteractions<F>;
    fn to_device_async(&self, stream: &CudaStream) -> Result<Self::DeviceType, CudaError> {
        Ok(DeviceInteractions {
            values_ptr: self.values_ptr.to_device_async(stream)?,
            values_col_weights_ptr: self.values_col_weights_ptr.to_device_async(stream)?,
            multiplicities_ptr: self.multiplicities_ptr.to_device_async(stream)?,
            values_col_weights: self.values_col_weights.to_device_async(stream)?,
            values_constants: self.values_constants.to_device_async(stream)?,
            mult_col_weights: self.mult_col_weights.to_device_async(stream)?,
            mult_constants: self.mult_constants.to_device_async(stream)?,
            arg_indices: self.arg_indices.to_device_async(stream)?,
            is_send: self.is_send.to_device_async(stream)?,
            num_interactions: self.num_interactions,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{device::memory::ToHost, matrix::RowMajorMatrixDevice, time::CudaInstant};

    use super::*;
    use p3_air::{Air, BaseAir};
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use p3_uni_stark::SymbolicAirBuilder;
    use rand::{thread_rng, Rng};

    use sp1_core_executor::Program;
    use sp1_core_machine::{memory::MemoryLocalChip, riscv::ByteChip};
    use sp1_stark::InteractionBuilder;

    const FIBONACCI_ELF: &[u8] =
        include_bytes!("../../../perf/programs/fibonacci/riscv32im-succinct-zkvm-elf");

    type F = BabyBear;
    const D: usize = 4;
    type EF = BinomialExtensionField<F, D>;

    fn test_chip<A>(chip: Chip<F, A>)
    where
        A: BaseAir<F>
            + MachineAir<F, Program = Program>
            + Air<InteractionBuilder<F>>
            + Air<SymbolicAirBuilder<F>>,
    {
        let mut rng = thread_rng();

        let program = Program::from(FIBONACCI_ELF).unwrap();
        let num_rows = 1 << 16;

        let preprocessed_trace = chip.generate_preprocessed_trace(&program);

        // Generate a random trace.
        let mut main_trace = RowMajorMatrix::<F>::rand(&mut rng, num_rows, chip.width());
        for val in main_trace.values.iter_mut() {
            *val = rng.gen::<F>();
        }

        // Transfer perm and main traces to the device.
        let prep_d = if let Some(preprocessed_trace) = preprocessed_trace.clone() {
            let prep_trace_d = preprocessed_trace.values.to_device().unwrap();
            let prep_d = RowMajorMatrixDevice::new(prep_trace_d, preprocessed_trace.width);
            let prep_d = prep_d.to_column_major();
            Some(prep_d)
        } else {
            None
        };

        let main_trace_d = main_trace.values.to_device().unwrap();
        let main_d = RowMajorMatrixDevice::new(main_trace_d, main_trace.width);
        let main_d = main_d.to_column_major();

        // Get randomness.
        let global_alpha = rng.gen::<EF>();
        let global_beta = rng.gen::<EF>();
        let local_alpha = rng.gen::<EF>();
        let local_beta = rng.gen::<EF>();

        let perm_generator = PermutationTraceGenerator::<F, EF, _>::default();
        // Generate the permutation rows on device.
        let time = CudaInstant::now().unwrap();
        let (perm_d, _) = perm_generator
            .generate_flattened_permutation_trace(
                &chip,
                prep_d.as_ref(),
                &main_d,
                &[global_alpha, global_beta, local_alpha, local_beta],
            )
            .unwrap();
        let elapsed = time.elapsed().unwrap();
        println!("Device generate_permutation_trace: {:?}", elapsed);

        let perm_h = perm_d.to_host();

        // print the dimensions
        println!("permutation trace: {:?}", perm_h.dimensions());

        let time = std::time::Instant::now();
        let expected_perm_trace = chip
            .generate_permutation_trace(
                preprocessed_trace.as_ref(),
                &main_trace,
                &[global_alpha, global_beta, local_alpha, local_beta],
            )
            .0
            .flatten_to_base::<F>();
        println!("Host generate_permutation_trace: {:?}", time.elapsed());

        // Compare the values to the host values.
        for (i, (exp, res)) in
            expected_perm_trace.values.iter().zip(perm_h.values.iter()).enumerate()
        {
            assert_eq!(exp, res, "at index {}", i);
        }
    }

    #[test]
    fn test_generate_flatenned_permutation_trace_device() {
        let memory_local_chip = Chip::new(MemoryLocalChip::new());
        test_chip(memory_local_chip);

        let byte_chip = Chip::new(ByteChip::<F>::default());
        test_chip(byte_chip);
    }
}
