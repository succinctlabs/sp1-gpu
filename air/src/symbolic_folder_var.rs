use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

use p3_field::AbstractField;

use crate::{operation::Operation, SymbolicFolderExpr, CUDA_P3_EVAL_CODE, EF, F};

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub enum SymbolicFolderVarType {
    Base = 0,
    Extension = 1,
    PreprocessedLocal = 2,
    PreprocessedNext = 3,
    MainLocal = 4,
    MainNext = 5,
    PermutationLocal = 6,
    PermutationNext = 7,
    PermutationChallenge = 8,
    CumulativeSum = 9,
    PublicValue = 10,
    IsFirstRow = 11,
    IsLastRow = 12,
    IsTransition = 13,
    Alpha = 14,
    Accumulator = 15,
    Empty = 16,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct SymbolicFolderVar {
    pub variant: SymbolicFolderVarType,
    pub f: F,
    pub ef: EF,
    pub idx: usize,
}

impl SymbolicFolderVar {
    pub fn empty() -> Self {
        Self { variant: SymbolicFolderVarType::Empty, f: F::zero(), ef: EF::zero(), idx: 0 }
    }

    pub fn base(f: F) -> Self {
        Self { variant: SymbolicFolderVarType::Base, f, ef: EF::zero(), idx: 0 }
    }

    pub fn extension(ef: EF) -> Self {
        Self { variant: SymbolicFolderVarType::Extension, f: F::zero(), ef, idx: 0 }
    }

    pub fn preprocessed_local(idx: usize) -> Self {
        Self {
            variant: SymbolicFolderVarType::PreprocessedLocal,
            f: F::zero(),
            ef: EF::zero(),
            idx,
        }
    }

    pub fn preprocessed_next(idx: usize) -> Self {
        Self { variant: SymbolicFolderVarType::PreprocessedNext, f: F::zero(), ef: EF::zero(), idx }
    }

    pub fn main_local(idx: usize) -> Self {
        Self { variant: SymbolicFolderVarType::MainLocal, f: F::zero(), ef: EF::zero(), idx }
    }

    pub fn main_next(idx: usize) -> Self {
        Self { variant: SymbolicFolderVarType::MainNext, f: F::zero(), ef: EF::zero(), idx }
    }

    pub fn permutation_local(idx: usize) -> Self {
        Self { variant: SymbolicFolderVarType::PermutationLocal, f: F::zero(), ef: EF::zero(), idx }
    }

    pub fn permutation_next(idx: usize) -> Self {
        Self { variant: SymbolicFolderVarType::PermutationNext, f: F::zero(), ef: EF::zero(), idx }
    }

    pub fn permutation_challenge(idx: usize) -> Self {
        Self {
            variant: SymbolicFolderVarType::PermutationChallenge,
            f: F::zero(),
            ef: EF::zero(),
            idx,
        }
    }

    pub fn cumulative_sum(idx: usize) -> Self {
        Self { variant: SymbolicFolderVarType::CumulativeSum, f: F::zero(), ef: EF::zero(), idx }
    }

    pub fn public_value(idx: usize) -> Self {
        Self { variant: SymbolicFolderVarType::PublicValue, f: F::zero(), ef: EF::zero(), idx }
    }

    pub fn is_first_row() -> Self {
        Self { variant: SymbolicFolderVarType::IsFirstRow, f: F::zero(), ef: EF::zero(), idx: 0 }
    }

    pub fn is_last_row() -> Self {
        Self { variant: SymbolicFolderVarType::IsLastRow, f: F::zero(), ef: EF::zero(), idx: 0 }
    }

    pub fn is_transition() -> Self {
        Self { variant: SymbolicFolderVarType::IsTransition, f: F::zero(), ef: EF::zero(), idx: 0 }
    }

    pub fn alpha() -> Self {
        Self { variant: SymbolicFolderVarType::Alpha, f: F::zero(), ef: EF::zero(), idx: 0 }
    }

    pub fn accumulator() -> Self {
        Self { variant: SymbolicFolderVarType::Accumulator, f: F::zero(), ef: EF::zero(), idx: 0 }
    }
}

impl From<SymbolicFolderVar> for SymbolicFolderExpr {
    fn from(value: SymbolicFolderVar) -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_v(output, value));
        drop(code);
        output
    }
}

impl Add<F> for SymbolicFolderVar {
    type Output = SymbolicFolderExpr;

    fn add(self, rhs: F) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::add_vf(output, self, rhs));
        drop(code);
        output
    }
}

impl Add<SymbolicFolderVar> for SymbolicFolderVar {
    type Output = SymbolicFolderExpr;

    fn add(self, rhs: SymbolicFolderVar) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::add_vv(output, self, rhs));
        drop(code);
        output
    }
}

impl Add<SymbolicFolderExpr> for SymbolicFolderVar {
    type Output = SymbolicFolderExpr;

    fn add(self, rhs: SymbolicFolderExpr) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::add_ve(output, self, rhs));
        drop(code);
        output
    }
}

impl Sub<F> for SymbolicFolderVar {
    type Output = SymbolicFolderExpr;

    fn sub(self, rhs: F) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::sub_vf(output, self, rhs));
        drop(code);
        output
    }
}

impl Sub<SymbolicFolderVar> for SymbolicFolderVar {
    type Output = SymbolicFolderExpr;

    fn sub(self, rhs: SymbolicFolderVar) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::sub_vv(output, self, rhs));
        drop(code);
        output
    }
}

impl Sub<SymbolicFolderExpr> for SymbolicFolderVar {
    type Output = SymbolicFolderExpr;

    fn sub(self, rhs: SymbolicFolderExpr) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::sub_ve(output, self, rhs));
        drop(code);
        output
    }
}

impl Mul<F> for SymbolicFolderVar {
    type Output = SymbolicFolderExpr;

    fn mul(self, rhs: F) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::mul_vf(output, self, rhs));
        drop(code);
        output
    }
}

impl Mul<SymbolicFolderVar> for SymbolicFolderVar {
    type Output = SymbolicFolderExpr;

    fn mul(self, rhs: SymbolicFolderVar) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::mul_vv(output, self, rhs));
        drop(code);
        output
    }
}

impl Mul<SymbolicFolderExpr> for SymbolicFolderVar {
    type Output = SymbolicFolderExpr;

    fn mul(self, rhs: SymbolicFolderExpr) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::mul_ve(output, self, rhs));
        drop(code);
        output
    }
}

impl Debug for SymbolicFolderVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.variant {
            SymbolicFolderVarType::Base => write!(f, "Base({:?})", self.idx),
            SymbolicFolderVarType::Extension => write!(f, "Extension({:?})", self.idx),
            SymbolicFolderVarType::PreprocessedLocal => {
                write!(f, "PreprocessedLocal({:?})", self.idx)
            }
            SymbolicFolderVarType::PreprocessedNext => {
                write!(f, "PreprocessedNext({:?})", self.idx)
            }
            SymbolicFolderVarType::MainLocal => write!(f, "MainLocal({:?})", self.idx),
            SymbolicFolderVarType::MainNext => write!(f, "MainNext({:?})", self.idx),
            SymbolicFolderVarType::PermutationLocal => {
                write!(f, "PermutationLocal({:?})", self.idx)
            }
            SymbolicFolderVarType::PermutationNext => write!(f, "PermutationNext({:?})", self.idx),
            SymbolicFolderVarType::PermutationChallenge => {
                write!(f, "PermutationChallenge({:?})", self.idx)
            }
            SymbolicFolderVarType::CumulativeSum => write!(f, "CumulativeSum({:?})", self.idx),
            SymbolicFolderVarType::PublicValue => write!(f, "PublicValue({:?})", self.idx),
            SymbolicFolderVarType::IsFirstRow => write!(f, "IsFirstRow({:?})", self.idx),
            SymbolicFolderVarType::IsLastRow => write!(f, "IsLastRow({:?})", self.idx),
            SymbolicFolderVarType::IsTransition => write!(f, "IsTransition({:?})", self.idx),
            SymbolicFolderVarType::Alpha => write!(f, "Alpha({:?})", self.idx),
            SymbolicFolderVarType::Accumulator => write!(f, "Accumulator({:?})", self.idx),
            SymbolicFolderVarType::Empty => write!(f, "Empty"),
        }
    }
}
