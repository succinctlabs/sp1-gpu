use std::fmt::Display;

use p3_air::BaseAir;
use p3_field::{ExtensionField, Field};
use sp1_stark::{air::MachineAir, Chip};

#[derive(Debug, Clone)]
pub struct ChipStatistics<F> {
    name: String,
    height: usize,
    preprocessed_cols: usize,
    main_cols: usize,
    permutation_cols: usize,
    quotient_cols: usize,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field> ChipStatistics<F> {
    pub fn new<EF: ExtensionField<F>, A: MachineAir<F>>(chip: &Chip<F, A>, height: usize) -> Self {
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
            _marker: std::marker::PhantomData,
        }
    }

    pub const fn total_width(&self) -> usize {
        self.preprocessed_cols + self.main_cols + self.permutation_cols + self.quotient_cols
    }

    /// Returns the size of all LDEs in bytes.
    pub fn lde_memory_size(&self, log_blowup: usize) -> usize {
        let blowup = 1 << log_blowup;

        self.total_width() * self.height * blowup * std::mem::size_of::<F>()
    }
}

impl<F: Field> Display for ChipStatistics<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<15} | Prep Cols = {:<5} | Main Cols = {:<5} | Perm Cols = {:<5} | Quot Cols = {:<5} | Rows = {:<5} | Cells = {:<10}",
            self.name,
            self.preprocessed_cols,
            self.main_cols,
            self.permutation_cols,
            self.quotient_cols,
            self.height,
            self.total_width() * self.height
        )
    }
}
