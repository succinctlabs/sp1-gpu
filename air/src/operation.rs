use std::fmt::Debug;

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
    AssignE = 3,

    AddVF = 4,
    AddVV = 5,
    AddVE = 6,
    AddEF = 7,
    AddEV = 8,
    AddEE = 9,
    AddAssignE = 10,

    SubVF = 11,
    SubVV = 12,
    SubVE = 13,
    SubEF = 14,
    SubEV = 15,
    SubEE = 16,
    SubAssignE = 17,

    MulVF = 18,
    MulVV = 19,
    MulVE = 20,
    MulEF = 21,
    MulEV = 22,
    MulEE = 23,
    MulAssignE = 24,
    MulAssignEF = 25,

    NegE = 26,
    Empty = 27,
}

#[derive(Clone, Copy)]
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

    pub fn assign_e(a: SymbolicFolderExpr, e: SymbolicFolderExpr) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::AssignE;
        op.a = a;
        op.b_expr = e;
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

    pub fn neg_e(a: SymbolicFolderExpr, b: SymbolicFolderExpr) -> Self {
        let mut op = Operation::empty();
        op.variant = OperationType::NegE;
        op.a = a;
        op.b_expr = b;
        op
    }
}

impl Debug for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.variant {
            OperationType::AssignF => write!(f, "AssignF({:?}, {:?})", self.a, self.b_f),
            OperationType::AssignEF => write!(f, "AssignEF({:?}, {:?})", self.a, self.b_ef),
            OperationType::AssignV => write!(f, "AssignV({:?}, {:?})", self.a, self.b_var),
            OperationType::AssignE => write!(f, "AssignE({:?}, {:?})", self.a, self.b_expr),

            OperationType::AddVF => {
                write!(f, "AddVF({:?}, {:?}, {:?})", self.a, self.b_var, self.c_f)
            }
            OperationType::AddVV => {
                write!(f, "AddVV({:?}, {:?}, {:?})", self.a, self.b_var, self.c_var)
            }
            OperationType::AddVE => {
                write!(f, "AddVE({:?}, {:?}, {:?})", self.a, self.b_var, self.c_expr)
            }
            OperationType::AddEF => {
                write!(f, "AddEF({:?}, {:?}, {:?})", self.a, self.b_expr, self.c_f)
            }
            OperationType::AddEV => {
                write!(f, "AddEV({:?}, {:?}, {:?})", self.a, self.b_expr, self.c_var)
            }
            OperationType::AddEE => {
                write!(f, "AddEE({:?}, {:?}, {:?})", self.a, self.b_expr, self.c_expr)
            }
            OperationType::AddAssignE => write!(f, "AddAssignE({:?}, {:?})", self.a, self.b_expr),

            OperationType::SubVF => {
                write!(f, "SubVF({:?}, {:?}, {:?})", self.a, self.b_var, self.c_f)
            }
            OperationType::SubVV => {
                write!(f, "SubVV({:?}, {:?}, {:?})", self.a, self.b_var, self.c_var)
            }
            OperationType::SubVE => {
                write!(f, "SubVE({:?}, {:?}, {:?})", self.a, self.b_var, self.c_expr)
            }
            OperationType::SubEF => {
                write!(f, "SubEF({:?}, {:?}, {:?})", self.a, self.b_expr, self.c_f)
            }
            OperationType::SubEV => {
                write!(f, "SubEV({:?}, {:?}, {:?})", self.a, self.b_expr, self.c_var)
            }
            OperationType::SubEE => {
                write!(f, "SubEE({:?}, {:?}, {:?})", self.a, self.b_expr, self.c_expr)
            }
            OperationType::SubAssignE => write!(f, "SubAssignE({:?}, {:?})", self.a, self.b_expr),

            OperationType::MulVF => {
                write!(f, "MulVF({:?}, {:?}, {:?})", self.a, self.b_var, self.c_f)
            }
            OperationType::MulVV => {
                write!(f, "MulVV({:?}, {:?}, {:?})", self.a, self.b_var, self.c_var)
            }
            OperationType::MulVE => {
                write!(f, "MulVE({:?}, {:?}, {:?})", self.a, self.b_var, self.c_expr)
            }
            OperationType::MulEF => {
                write!(f, "MulEF({:?}, {:?}, {:?})", self.a, self.b_expr, self.c_f)
            }
            OperationType::MulEV => {
                write!(f, "MulEV({:?}, {:?}, {:?})", self.a, self.b_expr, self.c_var)
            }
            OperationType::MulEE => {
                write!(f, "MulEE({:?}, {:?}, {:?})", self.a, self.b_expr, self.c_expr)
            }
            OperationType::MulAssignE => write!(f, "MulAssignE({:?}, {:?})", self.a, self.b_expr),
            OperationType::MulAssignEF => {
                write!(f, "MulAssignEF({:?}, {:?})", self.a, self.b_ef)
            }

            OperationType::NegE => write!(f, "NegE({:?})", self.a),
            OperationType::Empty => write!(f, "Empty"),
        }
    }
}
