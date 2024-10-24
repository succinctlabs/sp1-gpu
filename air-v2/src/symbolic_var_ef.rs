use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};
use tracing::instrument;

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
    #[instrument(skip_all, name = "Empty for SymbolicVarEF")]
    pub fn empty() -> Self {
        Self { variant: SymbolicVarEFKind::Empty, args: SymbolicVarEFArgs { empty: () } }
    }

    #[instrument(skip_all, name = "PermutationLocal for SymbolicVarEF")]
    pub fn permutation_local(idx: usize) -> Self {
        Self {
            variant: SymbolicVarEFKind::PermutationLocal,
            args: SymbolicVarEFArgs { idx: idx as u16 },
        }
    }

    #[instrument(skip_all, name = "PermutationNext for SymbolicVarEF")]
    pub fn permutation_next(idx: usize) -> Self {
        Self {
            variant: SymbolicVarEFKind::PermutationNext,
            args: SymbolicVarEFArgs { idx: idx as u16 },
        }
    }

    #[instrument(skip_all, name = "PermutationChallenge for SymbolicVarEF")]
    pub fn permutation_challenge(idx: u16) -> Self {
        Self { variant: SymbolicVarEFKind::PermutationChallenge, args: SymbolicVarEFArgs { idx } }
    }

    #[instrument(skip_all, name = "CumulativeSum for SymbolicVarEF")]
    pub fn cumulative_sum() -> Self {
        Self { variant: SymbolicVarEFKind::CumulativeSum, args: SymbolicVarEFArgs { empty: () } }
    }
}

impl From<SymbolicVarEF> for SymbolicExprEF {
    #[instrument(skip_all, name = "From<SymbolicVarEF> for SymbolicExprEF")]
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

    #[instrument(skip_all, name = "Add<EF> for SymbolicVarEF")]
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

    #[instrument(skip_all, name = "Add<SymbolicVarEF> for SymbolicVarEF")]
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

    #[instrument(skip_all, name = "Add<SymbolicExprEF> for SymbolicVarEF")]
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

    #[instrument(skip_all, name = "Sub<EF> for SymbolicVarEF")]
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

    #[instrument(skip_all, name = "Sub<SymbolicVarEF> for SymbolicVarEF")]
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

    #[instrument(skip_all, name = "Sub<SymbolicExprEF> for SymbolicVarEF")]
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

    #[instrument(skip_all, name = "Mul<EF> for SymbolicVarEF")]
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

    #[instrument(skip_all, name = "Mul<SymbolicVarEF> for SymbolicVarEF")]
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

    #[instrument(skip_all, name = "Mul<SymbolicExprEF> for SymbolicVarEF")]
    fn mul(self, rhs: SymbolicExprEF) -> Self::Output {
        let output = SymbolicExprEF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::e_mul_ve(output, self, rhs));
        drop(code);
        output
    }
}

impl Debug for SymbolicVarEF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            match self.variant {
                SymbolicVarEFKind::Empty => write!(f, "SymbolicVarEF::Empty"),
                SymbolicVarEFKind::Constant => {
                    write!(f, "SymbolicVarEF::Constant({})", self.args.ef)
                }
                SymbolicVarEFKind::PermutationLocal => {
                    write!(f, "SymbolicVarEF::PermutationLocal({})", self.args.idx)
                }
                SymbolicVarEFKind::PermutationNext => {
                    write!(f, "SymbolicVarEF::PermutationNext({})", self.args.idx)
                }
                SymbolicVarEFKind::PermutationChallenge => {
                    write!(f, "SymbolicVarEF::PermutationChallenge({})", self.args.idx)
                }
                SymbolicVarEFKind::CumulativeSum => write!(f, "SymbolicVarEF::CumulativeSum"),
            }
        }
    }
}
