use p3_air::{PairCol, VirtualPairCol};
use p3_baby_bear::BabyBear;
use p3_field::Field;
use sp1_core::lookup::Interaction;

use crate::device::{
    buffer::{DeviceBuffer, ToDevice},
    slice::DeviceSlice,
};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PairColDevice {
    column_idx: usize,
    is_preprocessed: bool,
}

#[derive(Debug)]
#[repr(C)]
pub struct DeviceInteractions<F: Field> {
    pub values_ptr: DeviceBuffer<usize>,
    pub multiplicities_ptr: DeviceBuffer<usize>,
    pub values_col_weights_ptr: DeviceBuffer<usize>,

    pub values_col_weights: DeviceBuffer<(PairColDevice, F)>,
    pub values_constants: DeviceBuffer<F>,

    pub mult_col_weights: DeviceBuffer<(PairColDevice, F)>,
    pub mult_constants: DeviceBuffer<F>,

    pub arg_indices: DeviceBuffer<usize>,
    pub is_send: DeviceBuffer<bool>,
    pub num_interactions: usize,
}

#[derive(Debug)]
#[repr(C)]
pub struct DeviceInteractionsView<'a, F: Field> {
    pub values_ptr: &'a DeviceSlice<usize>,
    pub multiplicities_ptr: &'a DeviceSlice<usize>,
    pub values_col_weights_ptr: &'a DeviceSlice<usize>,

    pub values_col_weights: &'a DeviceSlice<(PairColDevice, F)>,
    pub values_constants: &'a DeviceSlice<F>,

    pub mult_col_weights: &'a DeviceSlice<(PairColDevice, F)>,
    pub mult_constants: &'a DeviceSlice<F>,

    pub arg_indices: &'a DeviceSlice<usize>,
    pub is_send: &'a DeviceSlice<bool>,
    pub num_interactions: usize,
}

impl<F: Field> DeviceInteractions<F> {
    pub fn from_p3(sends: &[Interaction<F>], receives: &[Interaction<F>]) -> Self {
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
        for interaction in sends {
            // Register the values
            values_ptr.push(curr_values_ptr);
            for value in interaction.values.iter() {
                values_col_weights_ptr.push(curr_values_col_weight_ptr);
                for (col, weight) in value.column_weights.iter() {
                    let col = PairColDevice::from_p3(col);
                    values_col_weights.push((col, *weight));
                    curr_values_col_weight_ptr += 1;
                }
                values_constants.push(value.constant);
                curr_values_ptr += 1;
            }

            // Register the multiplicity values
            multiplicities_ptr.push(curr_mult_ptr);
            for (col, weight) in interaction.multiplicity.column_weights.iter() {
                let col = PairColDevice::from_p3(col);
                mult_col_weights.push((col, *weight));
                curr_mult_ptr += 1;
            }
            mult_constants.push(interaction.multiplicity.constant);

            arg_indices.push(interaction.argument_index());
            is_send.push(true);
        }

        Self {
            values_ptr: values_ptr.to_device(),
            values_col_weights_ptr: values_col_weights_ptr.to_device(),
            multiplicities_ptr: multiplicities_ptr.to_device(),
            values_col_weights: values_col_weights.to_device(),
            values_constants: values_constants.to_device(),
            mult_col_weights: mult_col_weights.to_device(),
            mult_constants: mult_constants.to_device(),
            arg_indices: arg_indices.to_device(),
            is_send: is_send.to_device(),
            num_interactions,
        }
    }

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

impl PairColDevice {
    pub fn from_p3(p3_pair_col: &PairCol) -> Self {
        match p3_pair_col {
            PairCol::Preprocessed(p3_pair_col) => Self {
                column_idx: *p3_pair_col,
                is_preprocessed: true,
            },
            PairCol::Main(p3_pair_col) => Self {
                column_idx: *p3_pair_col,
                is_preprocessed: false,
            },
        }
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct VirtualPairColDevice<F: Field> {
    columns: DeviceBuffer<PairColDevice>,
    weights: DeviceBuffer<F>,
    constant: F,
    n_columns: usize,
}

impl VirtualPairColDevice<BabyBear> {
    pub fn from_p3(c: &VirtualPairCol<BabyBear>) -> Self {
        let (columns, weights): (Vec<_>, Vec<_>) = c
            .column_weights
            .iter()
            .map(|cw| (PairColDevice::from_p3(&cw.0), cw.1))
            .unzip();
        let constant = c.constant;
        VirtualPairColDevice {
            columns: columns.to_device(),
            weights: weights.to_device(),
            constant,
            n_columns: columns.len(),
        }
    }

    pub fn view(&self) -> VirtualPairColViewDevice<BabyBear> {
        VirtualPairColViewDevice {
            columns: self.columns.as_ptr(),
            weights: self.weights.as_ptr(),
            constant: self.constant,
            n_columns: self.columns.len(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct VirtualPairColViewDevice<F: Field> {
    columns: *const PairColDevice,
    weights: *const F,
    constant: F,
    n_columns: usize,
}

#[derive(Debug)]
#[repr(C)]
pub struct InteractionDevice<F: Field> {
    pub values: DeviceBuffer<VirtualPairColViewDevice<F>>,
    pub multiplicity: VirtualPairColDevice<F>,
    pub kind: usize,
}

impl InteractionDevice<BabyBear> {
    pub fn from_p3(c: &Interaction<BabyBear>) -> (Self, Vec<VirtualPairColDevice<BabyBear>>) {
        let values = c
            .values
            .iter()
            .map(VirtualPairColDevice::from_p3)
            .collect::<Vec<_>>();
        let values_view = values.iter().map(|v| v.view()).collect::<Vec<_>>();
        let multiplicity = VirtualPairColDevice::from_p3(&c.multiplicity);
        let kind = c.argument_index();
        (
            InteractionDevice {
                values: values_view.to_device(),
                multiplicity,
                kind,
            },
            values,
        )
    }
}

#[derive(Debug)]
pub struct InteractionViewDevice<F: Field> {
    pub values: *const VirtualPairColViewDevice<F>,
    pub n_values: usize,
    pub multiplicity: VirtualPairColViewDevice<F>,
    pub kind: usize,
}

// fn generate_permutation_trace(
//     sends: &[Interaction<BabyBear>],
//     receives: &[Interaction<BabyBear>],
//     main: ColMajorMatrixDevice<BabyBear>,
//     random_elements: &[BinomialExtensionField<BabyBear, 4>],
//     batch_size: usize,
// ) {
//     let (sends_device, send_device_values): (Vec<_>, Vec<_>) =
//         sends.iter().map(|s| InteractionDevice::from_p3(s)).unzip();

//     let alphas = generate_interaction_rlc_elements(sends, receives, random_elements[0]);
//     let beta = random_elements[1];

//     let permutation_trace_width = permutation_trace_width(sends.len() + receives.len(), batch_size);
//     let height = main.height();

//     let mut permutation_trace = ColMajorMatrixDevice::new(
//         vec![BinomialExtensionField::<BabyBear, 4>::zero(); permutation_trace_width * height]
//             .to_device(),
//         height,
//     );
// }
