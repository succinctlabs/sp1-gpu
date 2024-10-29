#![allow(clippy::assign_op_pattern)]

pub mod instruction;
pub mod optimizer;
pub mod symbolic_expr_ef;
pub mod symbolic_expr_f;
pub mod symbolic_var_ef;
pub mod symbolic_var_f;

use std::sync::Mutex;

use instruction::{Instruction16, Instruction32};
use lazy_static::lazy_static;
use p3_air::BaseAir;
use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, ExtensionBuilder, PairBuilder,
    PermutationAirBuilder,
};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair};
use sp1_stark::{
    air::{EmptyMessageBuilder, MachineAir, MultiTableAirBuilder},
    Chip,
};
use sp1_stark::{AirOpenedValues, PROOF_MAX_NUM_PVS};
use symbolic_expr_ef::SymbolicExprEF;
use symbolic_expr_f::SymbolicExprF;
use symbolic_var_ef::SymbolicVarEF;
use symbolic_var_f::SymbolicVarF;

pub type F = BabyBear;

pub type EF = BinomialExtensionField<F, 4>;

lazy_static! {
    pub static ref CUDA_P3_EVAL_LOCK: Mutex<()> = Mutex::new(());
    pub static ref CUDA_P3_EVAL_CODE: Mutex<Vec<Instruction32>> = Mutex::new(Vec::new());
    pub static ref CUDA_P3_EVAL_F_CONSTANTS: Mutex<Vec<F>> = Mutex::new(Vec::new());
    pub static ref CUDA_P3_EVAL_EF_CONSTANTS: Mutex<Vec<EF>> = Mutex::new(Vec::new());
    pub static ref CUDA_P3_EVAL_EXPR_F_CTR: Mutex<u32> = Mutex::new(0);
    pub static ref CUDA_P3_EVAL_EXPR_EF_CTR: Mutex<u32> = Mutex::new(0);
}

pub struct SymbolicProverFolder<'a> {
    pub preprocessed:
        VerticalPair<RowMajorMatrixView<'a, SymbolicVarF>, RowMajorMatrixView<'a, SymbolicVarF>>,
    pub main:
        VerticalPair<RowMajorMatrixView<'a, SymbolicVarF>, RowMajorMatrixView<'a, SymbolicVarF>>,
    pub perm:
        VerticalPair<RowMajorMatrixView<'a, SymbolicVarEF>, RowMajorMatrixView<'a, SymbolicVarEF>>,
    pub perm_challenges: &'a [SymbolicVarEF],
    pub cumulative_sums: &'a [SymbolicVarEF],
    pub is_first_row: SymbolicVarF,
    pub is_last_row: SymbolicVarF,
    pub is_transition: SymbolicVarF,
    pub public_values: &'a [SymbolicVarF],
}

impl<'a> AirBuilder for SymbolicProverFolder<'a> {
    type F = F;
    type Var = SymbolicVarF;
    type Expr = SymbolicExprF;
    type M =
        VerticalPair<RowMajorMatrixView<'a, SymbolicVarF>, RowMajorMatrixView<'a, SymbolicVarF>>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row.into()
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row.into()
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition.into()
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: Self::Expr = x.into();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::f_assert_zero(x));
        drop(code);
    }
}

impl<'a> ExtensionBuilder for SymbolicProverFolder<'a> {
    type EF = EF;
    type ExprEF = SymbolicExprEF;
    type VarEF = SymbolicVarEF;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let x: SymbolicExprEF = x.into();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction32::e_assert_zero(x));
        drop(code);
    }
}

impl<'a> PermutationAirBuilder for SymbolicProverFolder<'a> {
    type MP =
        VerticalPair<RowMajorMatrixView<'a, SymbolicVarEF>, RowMajorMatrixView<'a, SymbolicVarEF>>;
    type RandomVar = SymbolicVarEF;

    fn permutation(&self) -> Self::MP {
        self.perm
    }
    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.perm_challenges
    }
}
impl<'a> MultiTableAirBuilder<'a> for SymbolicProverFolder<'a> {
    type Sum = SymbolicVarEF;

    fn cumulative_sums(&self) -> &'a [Self::Sum] {
        self.cumulative_sums
    }
}

impl<'a> PairBuilder for SymbolicProverFolder<'a> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed
    }
}

impl<'a> AirBuilderWithPublicValues for SymbolicProverFolder<'a> {
    type PublicVar = SymbolicVarF;

    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

impl<'a> EmptyMessageBuilder for SymbolicProverFolder<'a> {}

/// Generates code in CUDA for evaluating the constraint polynomial on the device.
pub fn codegen_cuda_eval<A>(chip: &Chip<F, A>) -> (Vec<Instruction16>, u32, u32, Vec<F>, Vec<EF>)
where
    A: for<'a> Air<SymbolicProverFolder<'a>> + MachineAir<F>,
{
    let preprocessed_width = chip.preprocessed_width() as u32;
    let width = chip.width() as u32;
    let permutation_width = chip.permutation_width() as u32;
    let preprocessed = AirOpenedValues {
        local: (0..preprocessed_width).map(SymbolicVarF::preprocessed_local).collect(),
        next: (0..preprocessed_width).map(SymbolicVarF::preprocessed_next).collect(),
    };
    let main = AirOpenedValues {
        local: (0..width).map(SymbolicVarF::main_local).collect(),
        next: (0..width).map(SymbolicVarF::main_next).collect(),
    };
    let perm = AirOpenedValues {
        local: (0..permutation_width).map(SymbolicVarEF::permutation_local).collect(),
        next: (0..permutation_width).map(SymbolicVarEF::permutation_next).collect(),
    };
    let public_values =
        (0..PROOF_MAX_NUM_PVS as u32).map(SymbolicVarF::public_value).collect::<Vec<_>>();
    let perm_challenges = (0..4).map(SymbolicVarEF::permutation_challenge).collect::<Vec<_>>();

    let mut folder = SymbolicProverFolder {
        preprocessed: preprocessed.view(),
        main: main.view(),
        perm: perm.view(),
        perm_challenges: &perm_challenges,
        cumulative_sums: &[SymbolicVarEF::cumulative_sum(0), SymbolicVarEF::cumulative_sum(1)],
        public_values: &public_values,
        is_first_row: SymbolicVarF::is_first_row(),
        is_last_row: SymbolicVarF::is_last_row(),
        is_transition: SymbolicVarF::is_transition(),
    };

    chip.eval(&mut folder);
    let code = CUDA_P3_EVAL_CODE.lock().unwrap().to_vec();
    let f_constants = CUDA_P3_EVAL_F_CONSTANTS.lock().unwrap().to_vec();
    let ef_constants = CUDA_P3_EVAL_EF_CONSTANTS.lock().unwrap().to_vec();

    CUDA_P3_EVAL_RESET();

    let (code, f_ctr, ef_ctr) = optimizer::optimize(code);

    (code, f_ctr as u32, ef_ctr as u32, f_constants, ef_constants)
}

#[allow(non_snake_case)]
pub fn CUDA_P3_EVAL_RESET() {
    *CUDA_P3_EVAL_CODE.lock().unwrap() = Vec::new();
    *CUDA_P3_EVAL_EF_CONSTANTS.lock().unwrap() = Vec::new();
    *CUDA_P3_EVAL_EXPR_F_CTR.lock().unwrap() = 0;
    *CUDA_P3_EVAL_EXPR_EF_CTR.lock().unwrap() = 0;
}

#[cfg(test)]
mod tests {

    use sp1_core_machine::{riscv::RiscvAir, utils::setup_logger};
    use sp1_stark::{air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2};

    use crate::codegen_cuda_eval;

    #[test]
    pub fn test_add() {
        setup_logger();

        let config = BabyBearPoseidon2::default();
        let machine = RiscvAir::machine(config);
        let chips = machine.chips();
        for chip in chips {
            if chip.name() == "AddSub" {
                let (code, f_ctr, _, f_constants, ef_constants) = codegen_cuda_eval(chip);
                println!("{:#?}", code);
                println!("{}", f_ctr);
                println!("{:?}", f_constants);
                println!("{:?}", ef_constants);
                return;
            }
        }
        panic!("no AddSub chip found");
    }
}
