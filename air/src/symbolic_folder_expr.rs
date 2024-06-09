use std::{
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use p3_field::AbstractField;

use crate::{
    operation::Operation, symbolic_folder_var::SymbolicFolderVar, CUDA_P3_EVAL_CODE,
    CUDA_P3_EVAL_EXPR_CTR, EF, F,
};

#[derive(Debug, Copy, Eq, PartialEq, Hash)]
#[repr(C)]
pub struct SymbolicFolderExpr(pub usize);

impl SymbolicFolderExpr {
    pub fn empty() -> Self {
        Self(usize::MAX)
    }

    pub fn alloc() -> Self {
        let mut tmp = CUDA_P3_EVAL_EXPR_CTR.lock().unwrap();
        let id = *tmp;
        *tmp += 1;
        drop(tmp);
        SymbolicFolderExpr(id)
    }
}

impl Default for SymbolicFolderExpr {
    fn default() -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::zero()));
        drop(code);
        output
    }
}

impl From<F> for SymbolicFolderExpr {
    fn from(value: F) -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, value));
        drop(code);
        output
    }
}

impl Add<F> for SymbolicFolderExpr {
    type Output = SymbolicFolderExpr;

    fn add(self, rhs: F) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::add_ef(output, self, rhs));
        drop(code);
        output
    }
}

impl Add<SymbolicFolderVar> for SymbolicFolderExpr {
    type Output = SymbolicFolderExpr;

    fn add(self, rhs: SymbolicFolderVar) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::add_ev(output, self, rhs));
        drop(code);
        output
    }
}

impl Add<SymbolicFolderExpr> for SymbolicFolderExpr {
    type Output = SymbolicFolderExpr;

    fn add(self, rhs: SymbolicFolderExpr) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::add_ee(output, self, rhs));
        drop(code);
        output
    }
}

impl AddAssign for SymbolicFolderExpr {
    fn add_assign(&mut self, rhs: SymbolicFolderExpr) {
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::add_assign_e(*self, rhs));
        drop(code);
    }
}

impl Sub<F> for SymbolicFolderExpr {
    type Output = SymbolicFolderExpr;

    fn sub(self, rhs: F) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::sub_ef(output, self, rhs));
        drop(code);
        output
    }
}

impl Sub<SymbolicFolderVar> for SymbolicFolderExpr {
    type Output = SymbolicFolderExpr;

    fn sub(self, rhs: SymbolicFolderVar) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::sub_ev(output, self, rhs));
        drop(code);
        output
    }
}

impl Sub<SymbolicFolderExpr> for SymbolicFolderExpr {
    type Output = SymbolicFolderExpr;

    fn sub(self, rhs: SymbolicFolderExpr) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::sub_ee(output, self, rhs));
        drop(code);
        output
    }
}

impl SubAssign for SymbolicFolderExpr {
    fn sub_assign(&mut self, rhs: SymbolicFolderExpr) {
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::sub_assign_e(*self, rhs));
        drop(code);
    }
}

impl Mul<F> for SymbolicFolderExpr {
    type Output = SymbolicFolderExpr;

    fn mul(self, rhs: F) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::mul_ef(output, self, rhs));
        drop(code);
        output
    }
}

impl Mul<SymbolicFolderVar> for SymbolicFolderExpr {
    type Output = SymbolicFolderExpr;

    fn mul(self, rhs: SymbolicFolderVar) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::mul_ev(output, self, rhs));
        drop(code);
        output
    }
}

impl Mul<SymbolicFolderExpr> for SymbolicFolderExpr {
    type Output = SymbolicFolderExpr;

    fn mul(self, rhs: SymbolicFolderExpr) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::mul_ee(output, self, rhs));
        drop(code);
        output
    }
}

impl MulAssign<EF> for SymbolicFolderExpr {
    fn mul_assign(&mut self, rhs: EF) {
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::mul_assign_ef(*self, rhs));
        drop(code);
    }
}

impl MulAssign<SymbolicFolderExpr> for SymbolicFolderExpr {
    fn mul_assign(&mut self, rhs: SymbolicFolderExpr) {
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::mul_assign_e(*self, rhs));
        drop(code);
    }
}

impl Neg for SymbolicFolderExpr {
    type Output = SymbolicFolderExpr;

    fn neg(self) -> Self::Output {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::neg_e(output));
        drop(code);
        output
    }
}

impl Sum for SymbolicFolderExpr {
    fn sum<I: Iterator<Item = SymbolicFolderExpr>>(iter: I) -> Self {
        let mut output = SymbolicFolderExpr::default();
        for item in iter {
            output += item;
        }
        output
    }
}

impl Product for SymbolicFolderExpr {
    fn product<I: Iterator<Item = SymbolicFolderExpr>>(iter: I) -> Self {
        let mut output = SymbolicFolderExpr::default();
        for item in iter {
            output *= item;
        }
        output
    }
}

impl Clone for SymbolicFolderExpr {
    #[allow(clippy::non_canonical_clone_impl)]
    fn clone(&self) -> Self {
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        let output = SymbolicFolderExpr::alloc();
        code.push(Operation::assign_e(output, *self));
        drop(code);
        output
    }
}

impl AbstractField for SymbolicFolderExpr {
    type F = EF;

    fn zero() -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::zero()));
        drop(code);
        output
    }

    fn one() -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::one()));
        drop(code);
        output
    }

    fn two() -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::two()));
        drop(code);
        output
    }

    fn neg_one() -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::neg_one()));
        drop(code);
        output
    }

    fn from_f(f: Self::F) -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_ef(output, f));
        drop(code);
        output
    }

    fn from_bool(b: bool) -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::from_bool(b)));
        drop(code);
        output
    }

    fn from_canonical_u8(n: u8) -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::from_canonical_u8(n)));
        drop(code);
        output
    }

    fn from_canonical_u16(n: u16) -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::from_canonical_u16(n)));
        drop(code);
        output
    }

    fn from_canonical_u32(n: u32) -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::from_canonical_u32(n)));
        drop(code);
        output
    }

    fn from_canonical_u64(n: u64) -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::from_canonical_u64(n)));
        drop(code);
        output
    }

    fn from_canonical_usize(n: usize) -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::from_canonical_usize(n)));
        drop(code);
        output
    }

    fn from_wrapped_u32(n: u32) -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::from_wrapped_u32(n)));
        drop(code);
        output
    }

    fn from_wrapped_u64(n: u64) -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::from_wrapped_u64(n)));
        drop(code);
        output
    }

    fn generator() -> Self {
        let output = SymbolicFolderExpr::alloc();
        let mut code = CUDA_P3_EVAL_CODE.lock().unwrap();
        code.push(Operation::assign_f(output, F::generator()));
        drop(code);
        output
    }
}
