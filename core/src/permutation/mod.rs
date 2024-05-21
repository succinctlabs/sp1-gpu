use p3_air::{PairCol, VirtualPairCol};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, Field};
use sp1_core::stark::permutation_trace_width;
use sp1_core::{lookup::Interaction, stark::generate_interaction_rlc_elements};

use crate::{
    device::buffer::{DeviceBuffer, ToDevice},
    matrix::ColMajorMatrixDevice,
};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PairColDevice {
    column_idx: usize,
    is_preprocessed: bool,
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

fn generate_permutation_trace(
    sends: &[Interaction<BabyBear>],
    receives: &[Interaction<BabyBear>],
    main: ColMajorMatrixDevice<BabyBear>,
    random_elements: &[BinomialExtensionField<BabyBear, 4>],
    batch_size: usize,
) {
    let (sends_device, send_device_values): (Vec<_>, Vec<_>) =
        sends.iter().map(|s| InteractionDevice::from_p3(s)).unzip();

    let alphas = generate_interaction_rlc_elements(sends, receives, random_elements[0]);
    let beta = random_elements[1];

    let permutation_trace_width = permutation_trace_width(sends.len() + receives.len(), batch_size);
    let height = main.height();

    let mut permutation_trace = ColMajorMatrixDevice::new(
        vec![BinomialExtensionField::<BabyBear, 4>::zero(); permutation_trace_width * height]
            .to_device(),
        height,
    );
}
