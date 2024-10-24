use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};
use tracing::instrument;

use crate::{instruction::Instruction, symbolic_expr_f::SymbolicExprF, CUDA_P3_EVAL_CODE, F};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct SymbolicVarF {
    pub variant: SymbolicVarFKind,
    pub args: SymbolicVarFArgs,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub enum SymbolicVarFKind {
    Empty = 0,
    Constant = 1,
    PreprocessedLocal = 2,
    PreprocessedNext = 3,
    MainLocal = 4,
    MainNext = 5,
    IsFirstRow = 6,
    IsLastRow = 7,
    IsTransition = 8,
    PublicValue = 9,
}

#[derive(Clone, Copy)]
pub union SymbolicVarFArgs {
    pub empty: (),
    pub f: F,
    pub idx: u16,
}

impl SymbolicVarF {
    pub fn empty() -> Self {
        Self { variant: SymbolicVarFKind::Empty, args: SymbolicVarFArgs { empty: () } }
    }

    pub fn constant(f: F) -> Self {
        Self { variant: SymbolicVarFKind::Constant, args: SymbolicVarFArgs { f } }
    }

    pub fn preprocessed_local(idx: usize) -> Self {
        Self {
            variant: SymbolicVarFKind::PreprocessedLocal,
            args: SymbolicVarFArgs { idx: idx as u16 },
        }
    }

    pub fn preprocessed_next(idx: usize) -> Self {
        Self {
            variant: SymbolicVarFKind::PreprocessedNext,
            args: SymbolicVarFArgs { idx: idx as u16 },
        }
    }

    pub fn main_local(idx: usize) -> Self {
        Self { variant: SymbolicVarFKind::MainLocal, args: SymbolicVarFArgs { idx: idx as u16 } }
    }

    pub fn main_next(idx: usize) -> Self {
        Self { variant: SymbolicVarFKind::MainNext, args: SymbolicVarFArgs { idx: idx as u16 } }
    }

    pub fn is_first_row() -> Self {
        Self { variant: SymbolicVarFKind::IsFirstRow, args: SymbolicVarFArgs { empty: () } }
    }

    pub fn is_last_row() -> Self {
        Self { variant: SymbolicVarFKind::IsLastRow, args: SymbolicVarFArgs { empty: () } }
    }

    pub fn is_transition() -> Self {
        Self { variant: SymbolicVarFKind::IsTransition, args: SymbolicVarFArgs { empty: () } }
    }

    pub fn public_value(idx: usize) -> Self {
        Self { variant: SymbolicVarFKind::PublicValue, args: SymbolicVarFArgs { idx: idx as u16 } }
    }
}

impl From<SymbolicVarF> for SymbolicExprF {
    #[instrument(skip_all, name = "From<SymbolicVarF> for SymbolicExprF")]
    fn from(val: SymbolicVarF) -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_assign_v(output, val));
        drop(code);
        output
    }
}

impl Add<F> for SymbolicVarF {
    type Output = SymbolicExprF;

    #[instrument(skip_all, name = "Add<F> for SymbolicVarF")]
    fn add(self, rhs: F) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_add_vc(output, self, rhs));
        drop(code);
        output
    }
}

impl Add<SymbolicVarF> for SymbolicVarF {
    type Output = SymbolicExprF;

    #[instrument(skip_all, name = "Add<SymbolicVarF> for SymbolicVarF")]
    fn add(self, rhs: SymbolicVarF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_add_vv(output, self, rhs));
        drop(code);
        output
    }
}

impl Add<SymbolicExprF> for SymbolicVarF {
    type Output = SymbolicExprF;

    #[instrument(skip_all, name = "Add<SymbolicExprF> for SymbolicVarF")]
    fn add(self, rhs: SymbolicExprF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_add_ve(output, self, rhs));
        drop(code);
        output
    }
}

impl Sub<F> for SymbolicVarF {
    type Output = SymbolicExprF;

    #[instrument(skip_all, name = "Sub<F> for SymbolicVarF")]
    fn sub(self, rhs: F) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_sub_vc(output, self, rhs));
        drop(code);
        output
    }
}

impl Sub<SymbolicVarF> for SymbolicVarF {
    type Output = SymbolicExprF;

    #[instrument(skip_all, name = "Sub<SymbolicVarF> for SymbolicVarF")]
    fn sub(self, rhs: SymbolicVarF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_sub_vv(output, self, rhs));
        drop(code);
        output
    }
}

impl Sub<SymbolicExprF> for SymbolicVarF {
    type Output = SymbolicExprF;

    #[instrument(skip_all, name = "Sub<SymbolicExprF> for SymbolicVarF")]
    fn sub(self, rhs: SymbolicExprF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_sub_ve(output, self, rhs));
        drop(code);
        output
    }
}

impl Mul<F> for SymbolicVarF {
    type Output = SymbolicExprF;

    #[instrument(skip_all, name = "Mul<F> for SymbolicVarF")]
    fn mul(self, rhs: F) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_mul_vc(output, self, rhs));
        drop(code);
        output
    }
}

impl Mul<SymbolicVarF> for SymbolicVarF {
    type Output = SymbolicExprF;

    #[instrument(skip_all, name = "Mul<SymbolicVarF> for SymbolicVarF")]
    fn mul(self, rhs: SymbolicVarF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_mul_vv(output, self, rhs));
        drop(code);
        output
    }
}

impl Mul<SymbolicExprF> for SymbolicVarF {
    type Output = SymbolicExprF;

    #[instrument(skip_all, name = "Mul<SymbolicExprF> for SymbolicVarF")]
    fn mul(self, rhs: SymbolicExprF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_mul_ve(output, self, rhs));
        drop(code);
        output
    }
}

impl Debug for SymbolicVarF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            match self.variant {
                SymbolicVarFKind::Empty => write!(f, "SymbolicVarF::Empty"),
                SymbolicVarFKind::Constant => write!(f, "SymbolicVarF::Constant({})", self.args.f),
                SymbolicVarFKind::PreprocessedLocal => {
                    write!(f, "SymbolicVarF::PreprocessedLocal({})", self.args.idx)
                }
                SymbolicVarFKind::PreprocessedNext => {
                    write!(f, "SymbolicVarF::PreprocessedNext({})", self.args.idx)
                }
                SymbolicVarFKind::MainLocal => {
                    write!(f, "SymbolicVarF::MainLocal({})", self.args.idx)
                }
                SymbolicVarFKind::MainNext => {
                    write!(f, "SymbolicVarF::MainNext({})", self.args.idx)
                }
                SymbolicVarFKind::IsFirstRow => write!(f, "SymbolicVarF::IsFirstRow"),
                SymbolicVarFKind::IsLastRow => write!(f, "SymbolicVarF::IsLastRow"),
                SymbolicVarFKind::IsTransition => write!(f, "SymbolicVarF::IsTransition"),
                SymbolicVarFKind::PublicValue => {
                    write!(f, "SymbolicVarF::PublicValue({})", self.args.idx)
                }
            }
        }
    }
}
