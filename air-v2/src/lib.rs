pub mod instruction;
// pub mod optimizer;
pub mod symbolic_expr_ef;
pub mod symbolic_expr_f;
pub mod symbolic_var_ef;
pub mod symbolic_var_f;

use std::sync::Mutex;

use instruction::Instruction;
use lazy_static::lazy_static;
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use sp1_stark::{
    air::MachineAir, AirOpenedValues, Chip, GenericVerifierConstraintFolder, PROOF_MAX_NUM_PVS,
};

/// The type of the field used for base elements.
pub type F = BabyBear;

/// The type of the field used for extension elements.
pub type EF = BinomialExtensionField<F, 4>;

lazy_static! {
    pub static ref CUDA_P3_EVAL_LOCK: Mutex<()> = Mutex::new(());
    pub static ref CUDA_P3_EVAL_CODE: Mutex<Vec<Instruction>> = Mutex::new(Vec::new());
    pub static ref CUDA_P3_EVAL_EXPR_F_CTR: Mutex<u32> = Mutex::new(0);
    pub static ref CUDA_P3_EVAL_EXPR_EF_CTR: Mutex<u32> = Mutex::new(0);
}

// pub struct SymbolicProverFolder<'a> {
//     pub preprocessed:
//         VerticalPair<RowMajorMatrixView<'a, SymbolicVarF>, RowMajorMatrixView<'a, SymbolicVarF>>,
//     pub main:
//         VerticalPair<RowMajorMatrixView<'a, SymbolicVarF>, RowMajorMatrixView<'a, SymbolicVarF>>,
//     pub perm:
//         VerticalPair<RowMajorMatrixView<'a, SymbolicVarEF>, RowMajorMatrixView<'a, SymbolicVarEF>>,
//     pub perm_challenges: &'a [SymbolicVarEF],
//     pub cumulative_sum: SymbolicVarEF,
//     pub is_first_row: SymbolicVarF,
//     pub is_last_row: SymbolicVarF,
//     pub is_transition: SymbolicVarF,
//     pub public_values: &'a [SymbolicVarF],
//     pub _marker: PhantomData<(F, EF)>,
// }

// /// Generates code in CUDA for evaluating the constraint polynomial on the device.
// pub fn codegen_cuda_eval<A>(chip: &Chip<F, A>) -> (Vec<Operation>, usize)
// where
//     A: for<'a> Air<P3EvalFolder<'a>> + MachineAir<F>,
// {
//     // Get a lock for compiling the folder, making sure that only one thread is compiling at a time.
//     let _guard = CUDA_P3_EVAL_LOCK.lock().unwrap();

//     CUDA_P3_EVAL_CODE_RESET();
//     CUDA_P3_EVAL_EXPR_CTR_RESET();

//     let preprocessed_width = chip.preprocessed_width();
//     let width = chip.width();
//     let permutation_width = chip.permutation_width();

//     let preprocessed = AirOpenedValues {
//         local: (0..preprocessed_width).map(SymbolicFolderVar::preprocessed_local).collect(),
//         next: (0..preprocessed_width).map(SymbolicFolderVar::preprocessed_next).collect(),
//     };
//     let main = AirOpenedValues {
//         local: (0..width).map(SymbolicFolderVar::main_local).collect(),
//         next: (0..width).map(SymbolicFolderVar::main_next).collect(),
//     };
//     let perm = AirOpenedValues {
//         local: (0..permutation_width).map(SymbolicFolderVar::permutation_local).collect(),
//         next: (0..permutation_width).map(SymbolicFolderVar::permutation_next).collect(),
//     };
//     let public_values =
//         (0..PROOF_MAX_NUM_PVS).map(SymbolicFolderVar::public_value).collect::<Vec<_>>();
//     let perm_challenges = (0..4).map(SymbolicFolderVar::permutation_challenge).collect::<Vec<_>>();

//     let accumulator = SymbolicFolderExpr::alloc();

//     let cumulative_sums = (0..2).map(SymbolicFolderVar::cumulative_sum).collect::<Vec<_>>();

//     let mut folder = P3EvalFolder {
//         preprocessed: preprocessed.view(),
//         main: main.view(),
//         perm: perm.view(),
//         perm_challenges: &perm_challenges,
//         cumulative_sums: &cumulative_sums,
//         public_values: &public_values,
//         is_first_row: SymbolicFolderVar::is_first_row(),
//         is_last_row: SymbolicFolderVar::is_last_row(),
//         is_transition: SymbolicFolderVar::is_transition(),
//         alpha: SymbolicFolderVar::alpha(),
//         accumulator,
//         _marker: PhantomData,
//     };
//     chip.eval(&mut folder);

//     let code = CUDA_P3_EVAL_CODE.lock().unwrap().clone();

//     CUDA_P3_EVAL_CODE_RESET();
//     CUDA_P3_EVAL_EXPR_CTR_RESET();

//     let (code, ctr) = optimizer::optimize(code);

//     (code, ctr)
// }

// /// Resets [CUDA_P3_EVAL_CODE] for the next compilation.
// #[allow(non_snake_case)]
// pub fn CUDA_P3_EVAL_CODE_RESET() {
//     *CUDA_P3_EVAL_CODE.lock().unwrap() = Vec::new();
// }

// /// Resets [CUDA_P3_EVAL_EXPR_CTR] for the next compilation.
// #[allow(non_snake_case)]
// pub fn CUDA_P3_EVAL_EXPR_CTR_RESET() {
//     *CUDA_P3_EVAL_EXPR_CTR.lock().unwrap() = 0;
// }

// #[cfg(test)]
// mod tests {

//     use sp1_core_machine::riscv::RiscvAir;
//     use sp1_stark::{air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2};

//     use crate::{codegen_cuda_eval, optimizer};

//     #[test]
//     pub fn test_add() {
//         let config = BabyBearPoseidon2::default();
//         let machine = RiscvAir::machine(config);
//         let chips = machine.chips();
//         for chip in chips {
//             if chip.name() == "AddSub" {
//                 let (code, _) = codegen_cuda_eval(chip);
//                 let code = optimizer::optimize(code);
//                 println!("{:?}", code);
//                 return;
//             }
//         }
//         panic!("no AddSub chip found");
//     }
// }
