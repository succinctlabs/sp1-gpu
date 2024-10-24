use std::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

use crate::{
    instruction::Instruction, symbolic_var_ef::SymbolicVarEF, symbolic_var_f::SymbolicVarF,
    CUDA_P3_EVAL_CODE, CUDA_P3_EVAL_EXPR_F_CTR, F,
};

use p3_field::AbstractField;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SymbolicExprF(pub u32);

impl SymbolicExprF {
    pub fn empty() -> Self {
        Self(u32::MAX)
    }

    pub fn alloc() -> Self {
        let mut tmp = CUDA_P3_EVAL_EXPR_F_CTR.lock().unwrap();
        let id = *tmp;
        *tmp += 1;
        drop(tmp);
        Self(id)
    }
}

impl Default for SymbolicExprF {
    fn default() -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_assign_c(output, F::zero()));
        drop(code);
        output
    }
}

impl From<F> for SymbolicExprF {
    fn from(f: F) -> Self {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_assign_c(output, f));
        drop(code);
        output
    }
}

impl Add<F> for SymbolicExprF {
    type Output = Self;

    fn add(self, rhs: F) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_add_ec(output, self, rhs));
        drop(code);
        output
    }
}

impl Add<SymbolicVarF> for SymbolicExprF {
    type Output = Self;

    fn add(self, rhs: SymbolicVarF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_add_ev(output, self, rhs));
        drop(code);
        output
    }
}

impl Add<SymbolicExprF> for SymbolicExprF {
    type Output = SymbolicExprF;

    fn add(self, rhs: SymbolicExprF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_add_ee(output, self, rhs));
        drop(code);
        output
    }
}

impl AddAssign<SymbolicExprF> for SymbolicExprF {
    fn add_assign(&mut self, rhs: SymbolicExprF) {
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_add_assign_e(*self, rhs));
        drop(code);
    }
}

impl Sub<F> for SymbolicExprF {
    type Output = Self;

    fn sub(self, rhs: F) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_sub_ec(output, self, rhs));
        drop(code);
        output
    }
}

impl Sub<SymbolicVarF> for SymbolicExprF {
    type Output = Self;

    fn sub(self, rhs: SymbolicVarF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_sub_ev(output, self, rhs));
        drop(code);
        output
    }
}

impl Sub<SymbolicExprF> for SymbolicExprF {
    type Output = Self;

    fn sub(self, rhs: SymbolicExprF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_sub_ee(output, self, rhs));
        drop(code);
        output
    }
}

impl SubAssign<SymbolicExprF> for SymbolicExprF {
    fn sub_assign(&mut self, rhs: SymbolicExprF) {
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_sub_assign_e(*self, rhs));
        drop(code);
    }
}

impl Mul<F> for SymbolicExprF {
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_mul_ec(output, self, rhs));
        drop(code);
        output
    }
}

impl Mul<SymbolicVarF> for SymbolicExprF {
    type Output = Self;

    fn mul(self, rhs: SymbolicVarF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_mul_ev(output, self, rhs));
        drop(code);
        output
    }
}

impl Mul<SymbolicExprF> for SymbolicExprF {
    type Output = Self;

    fn mul(self, rhs: SymbolicExprF) -> Self::Output {
        let output = SymbolicExprF::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_mul_ee(output, self, rhs));
        drop(code);
        output
    }
}

impl MulAssign<SymbolicExprF> for SymbolicExprF {
    fn mul_assign(&mut self, rhs: SymbolicExprF) {
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Instruction::f_mul_assign_e(*self, rhs));
        drop(code);
    }
}

impl Sum for SymbolicExprF {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut output = SymbolicExprF::default();
        for item in iter {
            output += item;
        }
        output
    }
}

impl Product for SymbolicExprF {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut output = SymbolicExprF::from(F::one());
        for item in iter {
            output *= item;
        }
        output
    }
}
