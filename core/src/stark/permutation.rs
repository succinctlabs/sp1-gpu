use std::{marker::PhantomData, ops::Mul};

use hashbrown::HashMap;
use p3_air::PairCol;
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractExtensionField, ExtensionField, Field};
use sp1_core::{
    air::{InteractionScope, MachineAir},
    lookup::Interaction,
    stark::{get_grouped_maps, Chip},
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
            Vec<BinomialExtensionField<BabyBear, 4>>,
        ),
        CudaError,
    > {
        let stream = main_trace.stream();
        const D: usize = 4;

        let batch_size = chip.logup_batch_size();
        let (grouped_sends, grouped_receives, grouped_widths) =
            get_grouped_maps(chip.sends(), chip.receives(), batch_size);

        let device_interactions = HostInteractions::new(grouped_sends, grouped_receives)
            .to_device_async(stream)
            .unwrap();

        let perm_width = chip.permutation_width();
        let height = main_trace.height;
        let mut perm_buffer =
            DeviceBuffer::<BabyBear>::with_capacity_in(perm_width * height * D, stream).unwrap();
        unsafe {
            perm_buffer.set_max_len();
        }
        let mut permutation_trace = ColMajorMatrixDevice::new(perm_buffer, height);

        let global_alpha = random_elements[0];
        let global_beta = random_elements[1];
        let local_alpha = random_elements[2];
        let local_beta = random_elements[3];

        let num_threads_per_block = 256;
        let num_blocks = height.div_ceil(num_threads_per_block);
        device_interactions.generate_flattened_permutation_trace(
            permutation_trace.view_mut(),
            preprocessed_trace
                .map(|mat| mat.view())
                .unwrap_or(MatrixViewDevice::null(false)),
            main_trace.view(),
            global_alpha,
            global_beta,
            local_alpha,
            local_beta,
            batch_size,
            num_blocks,
            num_threads_per_block,
            stream,
        )?;

        // Retrieve the cumulative sums.
        let row_idx = permutation_trace.height() - 1;
        let mut global_cumulative_sum = Default::default();
        let mut local_cumulative_sum = Default::default();
        for (scope, width) in grouped_widths {
            let (start_col_idx, scope_cumulative_sum) = match scope {
                InteractionScope::Global => ((width - 1) * D, &mut global_cumulative_sum),
                InteractionScope::Local => {
                    (permutation_trace.width() - D, &mut local_cumulative_sum)
                }
            };

            let cumulative_sum = BinomialExtensionField::<BabyBear, 4>::from_base_fn(|i| {
                let index = (start_col_idx + i) * permutation_trace.height() + row_idx;
                let val = permutation_trace.values[index..index + 1]
                    .as_host_vec(permutation_trace.stream());
                val[0]
            });

            *scope_cumulative_sum = cumulative_sum;
        }

        Ok((
            permutation_trace,
            vec![global_cumulative_sum, local_cumulative_sum],
        ))
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
    pub is_sends: Vec<bool>,
    pub is_globals: Vec<bool>,
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
    pub is_sends: DeviceBuffer<bool>,
    pub is_globals: DeviceBuffer<bool>,
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
    pub is_sends: *const bool,
    pub is_globals: *const bool,
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
    pub fn new(
        sends: HashMap<InteractionScope, Vec<Interaction<F>>>,
        receives: HashMap<InteractionScope, Vec<Interaction<F>>>,
    ) -> Self {
        let mut values_ptr = vec![];
        let mut values_col_weights_ptr = vec![];
        let mut multiplicities_ptr = vec![];
        let mut arg_indices = vec![];
        let mut is_sends = vec![];
        let mut is_globals = vec![];
        let mut mult_col_weights = vec![];
        let mut mult_constants = vec![];
        let mut values_col_weights = vec![];
        let mut values_constants = vec![];

        let num_interactions = sends.len() + receives.len();

        let mut curr_values_ptr = 0;
        let mut curr_values_col_weight_ptr = 0;
        let mut curr_mult_ptr = 0;

        // Put all of the interactions (for both global/local and send/receives) into a single list.
        // The ordering of the interactions is important to match with the CPU prover's ordering.
        // It should global sends, global receives, local sends, local receives.
        let interactions = [InteractionScope::Global, InteractionScope::Local]
            .map(|scope| {
                let empty_vec = vec![];
                let sends = sends
                    .get(&scope)
                    .unwrap_or(&empty_vec)
                    .clone()
                    .into_iter()
                    .map(move |i| (i, scope, true));
                let receives = receives
                    .get(&scope)
                    .unwrap_or(&empty_vec)
                    .clone()
                    .into_iter()
                    .map(move |i| (i, scope, false));

                sends.chain(receives)
            })
            .into_iter()
            .flatten();

        for (interaction, scope, is_send) in interactions {
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

            is_sends.push(is_send);
            is_globals.push(scope == InteractionScope::Global);
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
            is_sends,
            is_globals,

            num_interactions,
        }
    }

    pub fn populate_permutation_row<EF: ExtensionField<F>>(
        &self,
        row: &mut [EF],
        preprocessed_row: &[F],
        main_row: &[F],
        global_alpha: EF,
        global_beta: EF,
        local_alpha: EF,
        local_beta: EF,
        batch_size: usize,
    ) where
        F: Field,
    {
        for i in (0..self.num_interactions).step_by(batch_size) {
            let mut value = EF::zero();
            for j in 0..batch_size {
                // Calculate the interaction index.
                let index = i + j;

                if index >= self.num_interactions {
                    break;
                }

                // Initialize the denominator and beta powers.
                let mut denominator = if self.is_globals[index] {
                    global_alpha
                } else {
                    local_alpha
                };
                let mut beta_power = EF::one();

                // Add argument index to the denominator.
                let argument_index = self.arg_indices[index];
                denominator += beta_power * EF::from_base(argument_index);

                // Add the interaction values.
                for k in self.values_ptr[index]..self.values_ptr[index + 1] {
                    beta_power *= if self.is_globals[index] {
                        global_beta
                    } else {
                        local_beta
                    };
                    let mut acc = self.values_constants[k];
                    for l in self.values_col_weights_ptr[k]..self.values_col_weights_ptr[k + 1] {
                        acc += self.values_col_weights[l].get(preprocessed_row, main_row);
                    }
                    denominator += beta_power * acc;
                }

                // Calculate the multiplicity values.
                let is_send = self.is_sends[index];
                let mut mult = self.mult_constants[index];
                for k in self.multiplicities_ptr[index]..self.multiplicities_ptr[index + 1] {
                    mult += self.mult_col_weights[k].get(preprocessed_row, main_row);
                }

                if !is_send {
                    mult = -mult;
                }

                // Add `mult/ denominator` to the sum.
                value += EF::from_base(mult) / denominator;
            }
            // Assign the value to the row.
            let row_index = i / batch_size;
            row[row_index] = value;
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
            is_sends: self.is_sends.as_ptr(),
            is_globals: self.is_globals.as_ptr(),
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
        global_alpha: BinomialExtensionField<BabyBear, 4>,
        global_beta: BinomialExtensionField<BabyBear, 4>,
        local_alpha: BinomialExtensionField<BabyBear, 4>,
        local_beta: BinomialExtensionField<BabyBear, 4>,
        batch_size: usize,
        num_blocks: usize,
        num_threads_per_block: usize,
        stream: &CudaStream,
    ) {
        self.view().populate_permutation_rows_flattened(
            permutation,
            preprocessed,
            main,
            global_alpha,
            global_beta,
            local_alpha,
            local_beta,
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
        global_alpha: BinomialExtensionField<BabyBear, 4>,
        global_beta: BinomialExtensionField<BabyBear, 4>,
        local_alpha: BinomialExtensionField<BabyBear, 4>,
        local_beta: BinomialExtensionField<BabyBear, 4>,
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
            global_alpha,
            global_beta,
            local_alpha,
            local_beta,
            batch_size,
            num_blocks,
            num_threads_per_block,
            stream,
        );

        // Collect the cumulative sums using a scan in place.

        // TODO: optimize with a single kernel call instead of scan for each column of the batch.
        let col = permutation.width - D;
        let height = permutation.height;
        unsafe {
            for j in 0..4 {
                let last_col_ptr = permutation.values.add((col + j) * height);
                let cumulative_column = DeviceSlice::from_raw_parts_mut(last_col_ptr, height);
                cumulative_column.scan_inplace(stream)?;
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
        global_alpha: BinomialExtensionField<BabyBear, 4>,
        global_beta: BinomialExtensionField<BabyBear, 4>,
        local_alpha: BinomialExtensionField<BabyBear, 4>,
        local_beta: BinomialExtensionField<BabyBear, 4>,
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
                global_alpha,
                global_beta,
                local_alpha,
                local_beta,
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
            PairCol::Preprocessed(column_idx) => Self {
                column_idx,
                is_preprocessed: true,
                weight: F::one(),
            },
            PairCol::Main(column_idx) => Self {
                column_idx,
                is_preprocessed: false,
                weight: F::one(),
            },
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
            is_sends: self.is_sends.to_device_async(stream)?,
            is_globals: self.is_globals.to_device_async(stream)?,
            num_interactions: self.num_interactions,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::device::memory::ToHost;
    use crate::matrix::RowMajorMatrixDevice;
    use crate::time::CudaInstant;

    use super::*;
    use p3_air::BaseAir;
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::AbstractField;
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::thread_rng;
    use rand::Rng;

    use sp1_core::stark::permutation_trace_width;
    use sp1_core::utils::tests::FIBONACCI_ELF;

    use sp1_core::{
        air::MachineAir,
        runtime::Program,
        stark::{populate_permutation_row, ByteChip, Chip},
    };

    type F = BabyBear;
    const D: usize = 4;
    type EF = BinomialExtensionField<F, D>;

    // #[test]
    // fn test_populate_permutation_row_host() {
    //     let mut rng = thread_rng();

    //     let air = ByteChip::<F>::default();
    //     let chip = Chip::new(air);

    //     let program = Program::from(FIBONACCI_ELF);

    //     let num_rows = 1 << 16;
    //     let preprocessed_trace = chip.generate_preprocessed_trace(&program).unwrap();

    //     let batch_size = chip.logup_batch_size();
    //     let (grouped_sends, grouped_receives, grouped_widths) =
    //         get_grouped_maps(chip.sends(), chip.receives(), batch_size);

    //     // Generate a random trace.
    //     let main_trace = RowMajorMatrix::<F>::rand(&mut rng, num_rows, chip.width());

    //     // Get the host interactions.
    //     let host_interactions = HostInteractions::new(grouped_sends, grouped_receives);

    //     // For every row, compute the permutation row and compare the values.

    //     let batch_size = 2;
    //     let perm_width =
    //         permutation_trace_width(chip.sends().len() + chip.receives().len(), batch_size);
    //     let global_alpha = rng.gen::<EF>();
    //     let global_beta = rng.gen::<EF>();
    //     let local_alpha = rng.gen::<EF>();
    //     let local_beta = rng.gen::<EF>();
    //     for i in 0..num_rows {
    //         let prep_row = preprocessed_trace.row_slice(i);
    //         let main_row = main_trace.row_slice(i);

    //         let mut expected_row = vec![EF::zero(); perm_width];
    //         populate_permutation_row(
    //             &mut expected_row,
    //             &prep_row,
    //             &main_row,
    //             chip.sends(),
    //             chip.receives(),
    //             alpha,
    //             beta.powers(),

    //             batch_size,
    //         );

    //         let mut row = vec![EF::zero(); perm_width];
    //         host_interactions
    //             .populate_permutation_row(&mut row, &prep_row, &main_row, alpha, beta, batch_size);

    //         for (exp, val) in expected_row.iter().zip(row.iter()) {
    //             assert_eq!(exp, val, "row {} mismatch", i);
    //         }
    //     }
    // }

    #[test]
    fn test_generate_flatenned_permutation_trace_device() {
        let mut rng = thread_rng();

        let air = ByteChip::<F>::default();
        let chip = Chip::new(air);

        let program = Program::from(FIBONACCI_ELF);

        let num_rows = 1 << 16;
        let preprocessed_trace = chip.generate_preprocessed_trace(&program).unwrap();

        // Generate a random trace.
        let mut main_trace = RowMajorMatrix::<F>::rand(&mut rng, num_rows, chip.width());
        for val in main_trace.values.iter_mut() {
            *val = rng.gen::<F>();
        }

        // Transfer perm and main traces to the device.
        let prep_trace_d = preprocessed_trace.values.to_device().unwrap();
        let prep_d = RowMajorMatrixDevice::new(prep_trace_d, preprocessed_trace.width);
        let prep_d = prep_d.to_column_major();

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
                Some(&prep_d),
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
                Some(&preprocessed_trace),
                &main_trace,
                &[global_alpha, global_beta, local_alpha, local_beta],
            )
            .0
            .flatten_to_base::<F>();
        println!("Host generate_permutation_trace: {:?}", time.elapsed());

        // Compare the values to the host values.
        for (i, (exp, res)) in expected_perm_trace
            .values
            .iter()
            .zip(perm_h.values.iter())
            .enumerate()
        {
            assert_eq!(exp, res, "at index {}", i);
        }
    }
}
