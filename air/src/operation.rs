use p3_field::AbstractField;

use crate::{
    symbolic_folder_expr::SymbolicFolderExpr, symbolic_folder_var::SymbolicFolderVar, EF, F,
};

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub enum OperationType {
    AssignF = 0,
    AssignEF = 1,
    AssignV = 2,

    AddVF = 3,
    AddVV = 4,
    AddVE = 5,
    AddEF = 6,
    AddEV = 7,
    AddEE = 8,
    AddAssignE = 9,

    SubVF = 10,
    SubVV = 11,
    SubVE = 12,
    SubEF = 13,
    SubEV = 14,
    SubEE = 15,
    SubAssignE = 16,

    MulVF = 17,
    MulVV = 18,
    MulVE = 19,
    MulEF = 20,
    MulEV = 21,
    MulEE = 22,
    MulAssignE = 23,
    MulAssignEF = 24,

    NegE = 25,
    Empty = 26,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Operation {
    pub variant: OperationType,
    pub a: SymbolicFolderExpr,
    pub b_f: F,
    pub b_ef: EF,
    pub b_var: SymbolicFolderVar,
    pub b_expr: SymbolicFolderExpr,
    pub c_f: F,
    pub c_ef: EF,
    pub c_var: SymbolicFolderVar,
    pub c_expr: SymbolicFolderExpr,
}

impl Operation {
    pub fn empty() -> Self {
        Self {
            variant: OperationType::Empty,
            a: SymbolicFolderExpr::empty(),
            b_f: F::zero(),
            b_ef: EF::zero(),
            b_var: SymbolicFolderVar::empty(),
            b_expr: SymbolicFolderExpr::empty(),
            c_f: F::zero(),
            c_ef: EF::zero(),
            c_var: SymbolicFolderVar::empty(),
            c_expr: SymbolicFolderExpr::empty(),
        }
    }

    pub fn assign_f(a: SymbolicFolderExpr, f: F) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::AssignF;
        op.a = a;
        op.b_f = f;
        op
    }

    pub fn assign_ef(a: SymbolicFolderExpr, ef: EF) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::AssignEF;
        op.a = a;
        op.b_ef = ef;
        op
    }

    pub fn assign_v(a: SymbolicFolderExpr, var: SymbolicFolderVar) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::AssignV;
        op.a = a;
        op.b_var = var;
        op
    }

    pub fn add_vf(a: SymbolicFolderExpr, b: SymbolicFolderVar, c: F) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::AddVF;
        op.a = a;
        op.b_var = b;
        op.c_f = c;
        op
    }

    pub fn add_vv(a: SymbolicFolderExpr, b: SymbolicFolderVar, c: SymbolicFolderVar) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::AddVV;
        op.a = a;
        op.b_var = b;
        op.c_var = c;
        op
    }

    pub fn add_ve(a: SymbolicFolderExpr, b: SymbolicFolderVar, c: SymbolicFolderExpr) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::AddVE;
        op.a = a;
        op.b_var = b;
        op.c_expr = c;
        op
    }

    pub fn add_ef(a: SymbolicFolderExpr, b: SymbolicFolderExpr, c: F) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::AddEF;
        op.a = a;
        op.b_expr = b;
        op.c_f = c;
        op
    }

    pub fn add_ev(a: SymbolicFolderExpr, b: SymbolicFolderExpr, c: SymbolicFolderVar) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::AddEV;
        op.a = a;
        op.b_expr = b;
        op.c_var = c;
        op
    }

    pub fn add_ee(a: SymbolicFolderExpr, b: SymbolicFolderExpr, c: SymbolicFolderExpr) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::AddEE;
        op.a = a;
        op.b_expr = b;
        op.c_expr = c;
        op
    }

    pub fn add_assign_e(a: SymbolicFolderExpr, b: SymbolicFolderExpr) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::AddAssignE;
        op.a = a;
        op.b_expr = b;
        op
    }

    pub fn sub_vf(a: SymbolicFolderExpr, b: SymbolicFolderVar, c: F) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::SubVF;
        op.a = a;
        op.b_var = b;
        op.c_f = c;
        op
    }

    pub fn sub_vv(a: SymbolicFolderExpr, b: SymbolicFolderVar, c: SymbolicFolderVar) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::SubVV;
        op.a = a;
        op.b_var = b;
        op.c_var = c;
        op
    }

    pub fn sub_ve(a: SymbolicFolderExpr, b: SymbolicFolderVar, c: SymbolicFolderExpr) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::SubVE;
        op.a = a;
        op.b_var = b;
        op.c_expr = c;
        op
    }

    pub fn sub_ef(a: SymbolicFolderExpr, b: SymbolicFolderExpr, c: F) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::SubEF;
        op.a = a;
        op.b_expr = b;
        op.c_f = c;
        op
    }

    pub fn sub_ev(a: SymbolicFolderExpr, b: SymbolicFolderExpr, c: SymbolicFolderVar) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::SubEV;
        op.a = a;
        op.b_expr = b;
        op.c_var = c;
        op
    }

    pub fn sub_ee(a: SymbolicFolderExpr, b: SymbolicFolderExpr, c: SymbolicFolderExpr) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::SubEE;
        op.a = a;
        op.b_expr = b;
        op.c_expr = c;
        op
    }

    pub fn sub_assign_e(a: SymbolicFolderExpr, b: SymbolicFolderExpr) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::SubAssignE;
        op.a = a;
        op.b_expr = b;
        op
    }

    pub fn mul_vf(a: SymbolicFolderExpr, b: SymbolicFolderVar, c: F) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::MulVF;
        op.a = a;
        op.b_var = b;
        op.c_f = c;
        op
    }

    pub fn mul_vv(a: SymbolicFolderExpr, b: SymbolicFolderVar, c: SymbolicFolderVar) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::MulVV;
        op.a = a;
        op.b_var = b;
        op.c_var = c;
        op
    }

    pub fn mul_ve(a: SymbolicFolderExpr, b: SymbolicFolderVar, c: SymbolicFolderExpr) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::MulVE;
        op.a = a;
        op.b_var = b;
        op.c_expr = c;
        op
    }

    pub fn mul_ef(a: SymbolicFolderExpr, b: SymbolicFolderExpr, c: F) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::MulEF;
        op.a = a;
        op.b_expr = b;
        op.c_f = c;
        op
    }

    pub fn mul_ev(a: SymbolicFolderExpr, b: SymbolicFolderExpr, c: SymbolicFolderVar) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::MulEV;
        op.a = a;
        op.b_expr = b;
        op.c_var = c;
        op
    }

    pub fn mul_ee(a: SymbolicFolderExpr, b: SymbolicFolderExpr, c: SymbolicFolderExpr) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::MulEE;
        op.a = a;
        op.b_expr = b;
        op.c_expr = c;
        op
    }

    pub fn mul_assign_e(a: SymbolicFolderExpr, b: SymbolicFolderExpr) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::MulAssignE;
        op.a = a;
        op.b_expr = b;
        op
    }

    pub fn mul_assign_ef(a: SymbolicFolderExpr, b: EF) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::MulAssignEF;
        op.a = a;
        op.b_ef = b;
        op
    }

    pub fn neg_e(a: SymbolicFolderExpr) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::NegE;
        op.a = a;
        op
    }
}
