use std::ops::Mul;

use p3_air::PairCol;
use p3_field::{ExtensionField, Field};
use sp1_core::lookup::Interaction;

use crate::device::{
    buffer::{DeviceBuffer, ToDevice},
    slice::DeviceSlice,
};

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
    pub values_ptr: &'a DeviceSlice<usize>,
    pub multiplicities_ptr: &'a DeviceSlice<usize>,
    pub values_col_weights_ptr: &'a DeviceSlice<usize>,

    pub values_col_weights: &'a DeviceSlice<PairColDevice<F>>,
    pub values_constants: &'a DeviceSlice<F>,

    pub mult_col_weights: &'a DeviceSlice<PairColDevice<F>>,
    pub mult_constants: &'a DeviceSlice<F>,

    pub arg_indices: &'a DeviceSlice<F>,
    pub is_send: &'a DeviceSlice<bool>,
    pub num_interactions: usize,
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

        let num_sends = sends.len();
        for (i, interaction) in sends.iter().chain(receives).enumerate() {
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

            if i < num_sends {
                is_send.push(true);
            } else {
                is_send.push(false);
            }
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

    pub fn to_device(&self) -> DeviceInteractions<F> {
        DeviceInteractions {
            values_ptr: self.values_ptr.to_device(),
            values_col_weights_ptr: self.values_col_weights_ptr.to_device(),
            multiplicities_ptr: self.multiplicities_ptr.to_device(),
            values_col_weights: self.values_col_weights.to_device(),
            values_constants: self.values_constants.to_device(),
            mult_col_weights: self.mult_col_weights.to_device(),
            mult_constants: self.mult_constants.to_device(),
            arg_indices: self.arg_indices.to_device(),
            is_send: self.is_send.to_device(),
            num_interactions: self.num_interactions,
        }
    }

    pub fn populate_permutation_row<EF: ExtensionField<F>>(
        &self,
        row: &mut [EF],
        preprocessed_row: &[F],
        main_row: &[F],
        alpha: EF,
        beta: EF,
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
                let mut denominator = alpha;
                let mut beta_power = EF::one();

                // Add argument index to the denominator.
                let argument_index = self.arg_indices[index];
                denominator += beta_power * EF::from_base(argument_index);

                // Add the interaction values.
                for k in self.values_ptr[index]..self.values_ptr[index + 1] {
                    beta_power *= beta;
                    let mut acc = self.values_constants[k];
                    for l in self.values_col_weights_ptr[k]..self.values_col_weights_ptr[k + 1] {
                        acc += self.values_col_weights[l].get(preprocessed_row, main_row);
                    }
                    denominator += beta_power * acc;
                }

                // Calculate the multiplicity values.
                let is_send = self.is_send[index];
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
            values_ptr: self.values_ptr.as_slice(),
            multiplicities_ptr: self.multiplicities_ptr.as_slice(),
            values_col_weights_ptr: self.values_col_weights_ptr.as_slice(),

            values_col_weights: self.values_col_weights.as_slice(),
            values_constants: self.values_constants.as_slice(),

            mult_col_weights: self.mult_col_weights.as_slice(),
            mult_constants: self.mult_constants.as_slice(),

            arg_indices: self.arg_indices.as_slice(),
            is_send: self.is_send.as_slice(),
            num_interactions: self.num_interactions,
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

#[cfg(test)]
mod tests {
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

    #[test]
    fn test_populate_permutation_row() {
        let mut rng = thread_rng();

        let air = ByteChip::<F>::default();
        let chip = Chip::new(air);

        let program = Program::from(FIBONACCI_ELF);

        let num_rows = 1 << 16;
        let preprocessed_trace = chip.generate_preprocessed_trace(&program).unwrap();

        // Generate a random trace.
        let main_trace = RowMajorMatrix::<F>::rand(&mut rng, num_rows, chip.width());

        // Get the host interactions.
        let host_interactions = HostInteractions::new(chip.sends(), chip.receives());

        // For every row, compute the permutation row and compare the values.

        let batch_size = 2;
        let perm_width =
            permutation_trace_width(chip.sends().len() + chip.receives().len(), batch_size);
        let alpha = rng.gen::<EF>();
        let beta = rng.gen::<EF>();
        for i in 0..num_rows {
            let prep_row = preprocessed_trace.row_slice(i);
            let main_row = main_trace.row_slice(i);

            let mut expected_row = vec![EF::zero(); perm_width];
            populate_permutation_row(
                &mut expected_row,
                &prep_row,
                &main_row,
                chip.sends(),
                chip.receives(),
                alpha,
                beta.powers(),
                batch_size,
            );

            let mut row = vec![EF::zero(); perm_width];
            host_interactions
                .populate_permutation_row(&mut row, &prep_row, &main_row, alpha, beta, batch_size);

            for (exp, val) in expected_row.iter().zip(row.iter()) {
                assert_eq!(exp, val, "row {} mismatch", i);
            }
        }
    }
}
