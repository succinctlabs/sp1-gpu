use std::fmt::Display;

use p3_air::BaseAir;
use p3_field::{ExtensionField, Field};
use sp1_core::{air::MachineAir, stark::Chip};

#[derive(Debug, Clone)]
pub struct ChipStatistics {
    name: String,
    height: usize,
    preprocessed_cols: usize,
    main_cols: usize,
    permutation_cols: usize,
    quotient_cols: usize,
}

impl ChipStatistics {
    pub fn new<F: Field, EF: ExtensionField<F>, A: MachineAir<F>>(
        chip: &Chip<F, A>,
        height: usize,
    ) -> Self {
        let name = chip.name().to_owned();
        let preprocessed_cols = chip.preprocessed_width();
        let main_cols = chip.width();
        let permutation_cols = chip.permutation_width() * EF::D;
        let quotient_cols = chip.quotient_width() * EF::D;
        Self {
            name,
            height,
            preprocessed_cols,
            main_cols,
            permutation_cols,
            quotient_cols,
        }
    }
}

impl Display for ChipStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_width =
            self.preprocessed_cols + self.main_cols + self.permutation_cols + self.quotient_cols;
        write!(
            f,
            "{:<15} | Prep Cols = {:<5} | Main Cols = {:<5} | Perm Cols = {:<5} | Quot Cols = {:<5} | Rows = {:<5} | Cells = {:<10}",
            self.name,
            self.preprocessed_cols,
            self.main_cols,
            self.permutation_cols,
            self.quotient_cols,
            self.height,
            total_width * self.height
        )
    }
}
