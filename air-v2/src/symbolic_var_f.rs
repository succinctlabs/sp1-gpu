use std::ops::{Add, Mul, Sub};

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

    pub fn preprocessed_local(idx: u16) -> Self {
        Self { variant: SymbolicVarFKind::PreprocessedLocal, args: SymbolicVarFArgs { idx } }
    }

    pub fn preprocessed_next(idx: u16) -> Self {
        Self { variant: SymbolicVarFKind::PreprocessedNext, args: SymbolicVarFArgs { idx } }
    }

    pub fn main_local(idx: u16) -> Self {
        Self { variant: SymbolicVarFKind::MainLocal, args: SymbolicVarFArgs { idx } }
    }

    pub fn main_next(idx: u16) -> Self {
        Self { variant: SymbolicVarFKind::MainNext, args: SymbolicVarFArgs { idx } }
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
}

impl From<SymbolicVarF> for SymbolicExprF {
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

    fn mul(self, rhs: SymbolicExprF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_mul_ve(output, self, rhs));
        drop(code);
        output
    }
}
