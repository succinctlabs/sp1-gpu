use std::ops::{Add, Mul, Sub};

use crate::{instruction::Instruction, symbolic_expr_ef::SymbolicExprEF, CUDA_P3_EVAL_CODE, EF};

#[derive(Clone, Copy)]
#[repr(C)]
pub struct SymbolicVarEF {
    pub variant: SymbolicVarEFKind,
    pub args: SymbolicVarEFArgs,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub enum SymbolicVarEFKind {
    Empty = 0,
    Constant = 1,
    PermutationLocal = 2,
    PermutationNext = 3,
    PermutationChallenge = 4,
    CumulativeSum = 5,
}

#[derive(Clone, Copy)]
pub union SymbolicVarEFArgs {
    pub empty: (),
    pub ef: EF,
    pub idx: u16,
}

impl SymbolicVarEF {
    pub fn empty() -> Self {
        Self { variant: SymbolicVarEFKind::Empty, args: SymbolicVarEFArgs { empty: () } }
    }

    pub fn permutation_local(idx: u16) -> Self {
        Self { variant: SymbolicVarEFKind::PermutationLocal, args: SymbolicVarEFArgs { idx } }
    }

    pub fn permutation_next(idx: u16) -> Self {
        Self { variant: SymbolicVarEFKind::PermutationNext, args: SymbolicVarEFArgs { idx } }
    }

    pub fn permutation_challenge(idx: u16) -> Self {
        Self { variant: SymbolicVarEFKind::PermutationChallenge, args: SymbolicVarEFArgs { idx } }
    }

    pub fn cumulative_sum() -> Self {
        Self { variant: SymbolicVarEFKind::CumulativeSum, args: SymbolicVarEFArgs { empty: () } }
    }
}

impl From<SymbolicVarEF> for SymbolicExprEF {
    fn from(value: SymbolicVarEF) -> Self {
        let output = SymbolicExprEF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::e_assign_v(output, value));
        drop(code);
        output
    }
}

impl Add<EF> for SymbolicVarEF {
    type Output = SymbolicExprEF;

    fn add(self, rhs: EF) -> Self::Output {
        let output = SymbolicExprEF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::e_add_vc(output, self, rhs));
        drop(code);
        output
    }
}

impl Add<SymbolicVarEF> for SymbolicVarEF {
    type Output = SymbolicExprEF;

    fn add(self, rhs: SymbolicVarEF) -> Self::Output {
        let output = SymbolicExprEF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::e_add_vv(output, self, rhs));
        drop(code);
        output
    }
}

impl Add<SymbolicExprEF> for SymbolicVarEF {
    type Output = SymbolicExprEF;

    fn add(self, rhs: SymbolicExprEF) -> Self::Output {
        let output = SymbolicExprEF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::e_add_ve(output, self, rhs));
        drop(code);
        output
    }
}

impl Sub<EF> for SymbolicVarEF {
    type Output = SymbolicExprEF;

    fn sub(self, rhs: EF) -> Self::Output {
        let output = SymbolicExprEF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::e_sub_vc(output, self, rhs));
        drop(code);
        output
    }
}

impl Sub<SymbolicVarEF> for SymbolicVarEF {
    type Output = SymbolicExprEF;

    fn sub(self, rhs: SymbolicVarEF) -> Self::Output {
        let output = SymbolicExprEF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::e_sub_vv(output, self, rhs));
        drop(code);
        output
    }
}

impl Sub<SymbolicExprEF> for SymbolicVarEF {
    type Output = SymbolicExprEF;

    fn sub(self, rhs: SymbolicExprEF) -> Self::Output {
        let output = SymbolicExprEF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::e_sub_ve(output, self, rhs));
        drop(code);
        output
    }
}

impl Mul<EF> for SymbolicVarEF {
    type Output = SymbolicExprEF;

    fn mul(self, rhs: EF) -> Self::Output {
        let output = SymbolicExprEF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::e_mul_vc(output, self, rhs));
        drop(code);
        output
    }
}

impl Mul<SymbolicVarEF> for SymbolicVarEF {
    type Output = SymbolicExprEF;

    fn mul(self, rhs: SymbolicVarEF) -> Self::Output {
        let output = SymbolicExprEF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::e_mul_vv(output, self, rhs));
        drop(code);
        output
    }
}

impl Mul<SymbolicExprEF> for SymbolicVarEF {
    type Output = SymbolicExprEF;

    fn mul(self, rhs: SymbolicExprEF) -> Self::Output {
        let output = SymbolicExprEF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::e_mul_ve(output, self, rhs));
        drop(code);
        output
    }
}
