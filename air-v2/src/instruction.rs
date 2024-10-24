use std::fmt::Debug;
use std::mem::size_of;

use crate::{
    symbolic_expr_ef::SymbolicExprEF, symbolic_expr_f::SymbolicExprF,
    symbolic_var_ef::SymbolicVarEF, symbolic_var_f::SymbolicVarF, EF, F,
};

pub const INSTRUCTION_SIZE: usize = size_of::<Instruction>();

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Instruction {
    pub opcode: Opcode,
    pub args: Arguments,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub enum Opcode {
    Empty = 0,

    FAssignC = 1,
    FAssignV = 2,
    FAssignE = 3,

    FAddVC = 4,
    FAddVV = 5,
    FAddVE = 6,

    FAddEC = 7,
    FAddEV = 8,
    FAddEE = 9,
    FAddAssignE = 10,

    FSubVC = 11,
    FSubVV = 12,
    FSubVE = 13,

    FSubEC = 14,
    FSubEV = 15,
    FSubEE = 16,
    FSubAssignE = 17,

    FMulVC = 18,
    FMulVV = 19,
    FMulVE = 20,

    FMulEC = 21,
    FMulEV = 22,
    FMulEE = 23,
    FMulAssignE = 24,

    EAssignC = 25,
    EAssignV = 26,
    EAssignE = 27,

    EAddVC = 28,
    EAddVV = 29,
    EAddVE = 30,

    EAddEC = 31,
    EAddEV = 32,
    EAddEE = 33,
    EAddAssignE = 34,

    ESubVC = 35,
    ESubVV = 36,
    ESubVE = 37,

    ESubEC = 38,
    ESubEV = 39,
    ESubEE = 40,
    ESubAssignE = 41,

    EMulVC = 42,
    EMulVV = 43,
    EMulVE = 44,

    EMulEC = 45,
    EMulEV = 46,
    EMulEE = 47,
    EMulAssignE = 48,
}

#[derive(Clone, Copy)]
pub union Arguments {
    pub empty: (),

    pub f_op_c: FOperationC,
    pub f_op_v: FOperationV,
    pub f_op_e: FOperationE,

    pub f_op_vc: FOperationVC,
    pub f_op_vv: FOperationVV,
    pub f_op_ve: FOperationVE,

    pub f_op_ec: FOperationEC,
    pub f_op_ev: FOperationEV,
    pub f_op_ee: FOperationEE,

    pub e_op_c: EOperationC,
    pub e_op_v: EOperationV,
    pub e_op_e: EOperationE,

    pub e_op_vc: EOperationVC,
    pub e_op_vv: EOperationVV,
    pub e_op_ve: EOperationVE,

    pub e_op_ec: EOperationEC,
    pub e_op_ev: EOperationEV,
    pub e_op_ee: EOperationEE,
}

#[derive(Clone, Copy)]
pub struct FOperationC {
    pub a: SymbolicExprF,
    pub b: F,
}

#[derive(Clone, Copy)]
pub struct FOperationV {
    pub a: SymbolicExprF,
    pub b: SymbolicVarF,
}

#[derive(Clone, Copy)]
pub struct FOperationE {
    pub a: SymbolicExprF,
    pub b: SymbolicExprF,
}

#[derive(Clone, Copy)]
pub struct FOperationVC {
    pub a: SymbolicExprF,
    pub b: SymbolicVarF,
    pub c: F,
}

#[derive(Clone, Copy)]
pub struct FOperationVV {
    pub a: SymbolicExprF,
    pub b: SymbolicVarF,
    pub c: SymbolicVarF,
}

#[derive(Clone, Copy)]
pub struct FOperationVE {
    pub a: SymbolicExprF,
    pub b: SymbolicVarF,
    pub c: SymbolicExprF,
}

#[derive(Clone, Copy)]
pub struct FOperationEC {
    pub a: SymbolicExprF,
    pub b: SymbolicExprF,
    pub c: F,
}

#[derive(Clone, Copy)]
pub struct FOperationEV {
    pub a: SymbolicExprF,
    pub b: SymbolicExprF,
    pub c: SymbolicVarF,
}

#[derive(Clone, Copy)]
pub struct FOperationEE {
    pub a: SymbolicExprF,
    pub b: SymbolicExprF,
    pub c: SymbolicExprF,
}

#[derive(Clone, Copy)]
pub struct EOperationC {
    pub a: SymbolicExprEF,
    pub b: EF,
}

#[derive(Clone, Copy)]
pub struct EOperationV {
    pub a: SymbolicExprEF,
    pub b: SymbolicVarEF,
}

#[derive(Clone, Copy)]
pub struct EOperationE {
    pub a: SymbolicExprEF,
    pub b: SymbolicExprEF,
}

#[derive(Clone, Copy)]
pub struct EOperationVC {
    pub a: SymbolicExprEF,
    pub b: SymbolicVarEF,
    pub c: EF,
}

#[derive(Clone, Copy)]
pub struct EOperationVV {
    pub a: SymbolicExprEF,
    pub b: SymbolicVarEF,
    pub c: SymbolicVarEF,
}

#[derive(Clone, Copy)]
pub struct EOperationVE {
    pub a: SymbolicExprEF,
    pub b: SymbolicVarEF,
    pub c: SymbolicExprEF,
}

#[derive(Clone, Copy)]
pub struct EOperationEC {
    pub a: SymbolicExprEF,
    pub b: SymbolicExprEF,
    pub c: EF,
}

#[derive(Clone, Copy)]
pub struct EOperationEV {
    pub a: SymbolicExprEF,
    pub b: SymbolicExprEF,
    pub c: SymbolicVarEF,
}

#[derive(Clone, Copy)]
pub struct EOperationEE {
    pub a: SymbolicExprEF,
    pub b: SymbolicExprEF,
    pub c: SymbolicExprEF,
}

impl Instruction {
    pub fn f_assign_c(a: SymbolicExprF, b: F) -> Self {
        Self { opcode: Opcode::FAssignC, args: Arguments { f_op_c: FOperationC { a, b } } }
    }

    pub fn f_assign_v(a: SymbolicExprF, b: SymbolicVarF) -> Self {
        Self { opcode: Opcode::FAssignV, args: Arguments { f_op_v: FOperationV { a, b } } }
    }

    pub fn f_assign_e(a: SymbolicExprF, b: SymbolicExprF) -> Self {
        Self { opcode: Opcode::FAssignE, args: Arguments { f_op_e: FOperationE { a, b } } }
    }

    pub fn f_add_vc(a: SymbolicExprF, b: SymbolicVarF, c: F) -> Self {
        Self { opcode: Opcode::FAddVC, args: Arguments { f_op_vc: FOperationVC { a, b, c } } }
    }

    pub fn f_add_vv(a: SymbolicExprF, b: SymbolicVarF, c: SymbolicVarF) -> Self {
        Self { opcode: Opcode::FAddVV, args: Arguments { f_op_vv: FOperationVV { a, b, c } } }
    }

    pub fn f_add_ve(a: SymbolicExprF, b: SymbolicVarF, c: SymbolicExprF) -> Self {
        Self { opcode: Opcode::FAddVE, args: Arguments { f_op_ve: FOperationVE { a, b, c } } }
    }

    pub fn f_add_ec(a: SymbolicExprF, b: SymbolicExprF, c: F) -> Self {
        Self { opcode: Opcode::FAddEC, args: Arguments { f_op_ec: FOperationEC { a, b, c } } }
    }

    pub fn f_add_ev(a: SymbolicExprF, b: SymbolicExprF, c: SymbolicVarF) -> Self {
        Self { opcode: Opcode::FAddEV, args: Arguments { f_op_ev: FOperationEV { a, b, c } } }
    }

    pub fn f_add_ee(a: SymbolicExprF, b: SymbolicExprF, c: SymbolicExprF) -> Self {
        Self { opcode: Opcode::FAddEE, args: Arguments { f_op_ee: FOperationEE { a, b, c } } }
    }

    pub fn f_add_assign_e(a: SymbolicExprF, b: SymbolicExprF) -> Self {
        Self { opcode: Opcode::FAddAssignE, args: Arguments { f_op_e: FOperationE { a, b } } }
    }

    pub fn f_sub_vc(a: SymbolicExprF, b: SymbolicVarF, c: F) -> Self {
        Self { opcode: Opcode::FSubVC, args: Arguments { f_op_vc: FOperationVC { a, b, c } } }
    }

    pub fn f_sub_vv(a: SymbolicExprF, b: SymbolicVarF, c: SymbolicVarF) -> Self {
        Self { opcode: Opcode::FSubVV, args: Arguments { f_op_vv: FOperationVV { a, b, c } } }
    }

    pub fn f_sub_ve(a: SymbolicExprF, b: SymbolicVarF, c: SymbolicExprF) -> Self {
        Self { opcode: Opcode::FSubVE, args: Arguments { f_op_ve: FOperationVE { a, b, c } } }
    }

    pub fn f_sub_ec(a: SymbolicExprF, b: SymbolicExprF, c: F) -> Self {
        Self { opcode: Opcode::FSubEC, args: Arguments { f_op_ec: FOperationEC { a, b, c } } }
    }

    pub fn f_sub_ev(a: SymbolicExprF, b: SymbolicExprF, c: SymbolicVarF) -> Self {
        Self { opcode: Opcode::FSubEV, args: Arguments { f_op_ev: FOperationEV { a, b, c } } }
    }

    pub fn f_sub_ee(a: SymbolicExprF, b: SymbolicExprF, c: SymbolicExprF) -> Self {
        Self { opcode: Opcode::FSubEE, args: Arguments { f_op_ee: FOperationEE { a, b, c } } }
    }

    pub fn f_sub_assign_e(a: SymbolicExprF, b: SymbolicExprF) -> Self {
        Self { opcode: Opcode::FSubAssignE, args: Arguments { f_op_e: FOperationE { a, b } } }
    }

    pub fn f_mul_vc(a: SymbolicExprF, b: SymbolicVarF, c: F) -> Self {
        Self { opcode: Opcode::FMulVC, args: Arguments { f_op_vc: FOperationVC { a, b, c } } }
    }

    pub fn f_mul_vv(a: SymbolicExprF, b: SymbolicVarF, c: SymbolicVarF) -> Self {
        Self { opcode: Opcode::FMulVV, args: Arguments { f_op_vv: FOperationVV { a, b, c } } }
    }

    pub fn f_mul_ve(a: SymbolicExprF, b: SymbolicVarF, c: SymbolicExprF) -> Self {
        Self { opcode: Opcode::FMulVE, args: Arguments { f_op_ve: FOperationVE { a, b, c } } }
    }

    pub fn f_mul_ec(a: SymbolicExprF, b: SymbolicExprF, c: F) -> Self {
        Self { opcode: Opcode::FMulEC, args: Arguments { f_op_ec: FOperationEC { a, b, c } } }
    }

    pub fn f_mul_ev(a: SymbolicExprF, b: SymbolicExprF, c: SymbolicVarF) -> Self {
        Self { opcode: Opcode::FMulEV, args: Arguments { f_op_ev: FOperationEV { a, b, c } } }
    }

    pub fn f_mul_ee(a: SymbolicExprF, b: SymbolicExprF, c: SymbolicExprF) -> Self {
        Self { opcode: Opcode::FMulEE, args: Arguments { f_op_ee: FOperationEE { a, b, c } } }
    }

    pub fn f_mul_assign_e(a: SymbolicExprF, b: SymbolicExprF) -> Self {
        Self { opcode: Opcode::FMulAssignE, args: Arguments { f_op_e: FOperationE { a, b } } }
    }

    pub fn e_assign_c(a: SymbolicExprEF, b: EF) -> Self {
        Self { opcode: Opcode::EAssignC, args: Arguments { e_op_c: EOperationC { a, b } } }
    }

    pub fn e_assign_v(a: SymbolicExprEF, b: SymbolicVarEF) -> Self {
        Self { opcode: Opcode::EAssignV, args: Arguments { e_op_v: EOperationV { a, b } } }
    }

    pub fn e_assign_e(a: SymbolicExprEF, b: SymbolicExprEF) -> Self {
        Self { opcode: Opcode::EAssignE, args: Arguments { e_op_e: EOperationE { a, b } } }
    }

    pub fn e_add_vc(a: SymbolicExprEF, b: SymbolicVarEF, c: EF) -> Self {
        Self { opcode: Opcode::EAddVC, args: Arguments { e_op_vc: EOperationVC { a, b, c } } }
    }

    pub fn e_add_vv(a: SymbolicExprEF, b: SymbolicVarEF, c: SymbolicVarEF) -> Self {
        Self { opcode: Opcode::EAddVV, args: Arguments { e_op_vv: EOperationVV { a, b, c } } }
    }

    pub fn e_add_ve(a: SymbolicExprEF, b: SymbolicVarEF, c: SymbolicExprEF) -> Self {
        Self { opcode: Opcode::EAddVE, args: Arguments { e_op_ve: EOperationVE { a, b, c } } }
    }

    pub fn e_add_ec(a: SymbolicExprEF, b: SymbolicExprEF, c: EF) -> Self {
        Self { opcode: Opcode::EAddEC, args: Arguments { e_op_ec: EOperationEC { a, b, c } } }
    }

    pub fn e_add_ev(a: SymbolicExprEF, b: SymbolicExprEF, c: SymbolicVarEF) -> Self {
        Self { opcode: Opcode::EAddEV, args: Arguments { e_op_ev: EOperationEV { a, b, c } } }
    }

    pub fn e_add_ee(a: SymbolicExprEF, b: SymbolicExprEF, c: SymbolicExprEF) -> Self {
        Self { opcode: Opcode::EAddEE, args: Arguments { e_op_ee: EOperationEE { a, b, c } } }
    }

    pub fn e_add_assign_e(a: SymbolicExprEF, b: SymbolicExprEF) -> Self {
        Self { opcode: Opcode::EAddAssignE, args: Arguments { e_op_e: EOperationE { a, b } } }
    }

    pub fn e_sub_vc(a: SymbolicExprEF, b: SymbolicVarEF, c: EF) -> Self {
        Self { opcode: Opcode::ESubVC, args: Arguments { e_op_vc: EOperationVC { a, b, c } } }
    }

    pub fn e_sub_vv(a: SymbolicExprEF, b: SymbolicVarEF, c: SymbolicVarEF) -> Self {
        Self { opcode: Opcode::ESubVV, args: Arguments { e_op_vv: EOperationVV { a, b, c } } }
    }

    pub fn e_sub_ve(a: SymbolicExprEF, b: SymbolicVarEF, c: SymbolicExprEF) -> Self {
        Self { opcode: Opcode::ESubVE, args: Arguments { e_op_ve: EOperationVE { a, b, c } } }
    }

    pub fn e_sub_ec(a: SymbolicExprEF, b: SymbolicExprEF, c: EF) -> Self {
        Self { opcode: Opcode::ESubEC, args: Arguments { e_op_ec: EOperationEC { a, b, c } } }
    }

    pub fn e_sub_ev(a: SymbolicExprEF, b: SymbolicExprEF, c: SymbolicVarEF) -> Self {
        Self { opcode: Opcode::ESubEV, args: Arguments { e_op_ev: EOperationEV { a, b, c } } }
    }

    pub fn e_sub_ee(a: SymbolicExprEF, b: SymbolicExprEF, c: SymbolicExprEF) -> Self {
        Self { opcode: Opcode::ESubEE, args: Arguments { e_op_ee: EOperationEE { a, b, c } } }
    }

    pub fn e_sub_assign_e(a: SymbolicExprEF, b: SymbolicExprEF) -> Self {
        Self { opcode: Opcode::ESubAssignE, args: Arguments { e_op_e: EOperationE { a, b } } }
    }

    pub fn e_mul_vc(a: SymbolicExprEF, b: SymbolicVarEF, c: EF) -> Self {
        Self { opcode: Opcode::EMulVC, args: Arguments { e_op_vc: EOperationVC { a, b, c } } }
    }

    pub fn e_mul_vv(a: SymbolicExprEF, b: SymbolicVarEF, c: SymbolicVarEF) -> Self {
        Self { opcode: Opcode::EMulVV, args: Arguments { e_op_vv: EOperationVV { a, b, c } } }
    }

    pub fn e_mul_ve(a: SymbolicExprEF, b: SymbolicVarEF, c: SymbolicExprEF) -> Self {
        Self { opcode: Opcode::EMulVE, args: Arguments { e_op_ve: EOperationVE { a, b, c } } }
    }

    pub fn e_mul_ec(a: SymbolicExprEF, b: SymbolicExprEF, c: EF) -> Self {
        Self { opcode: Opcode::EMulEC, args: Arguments { e_op_ec: EOperationEC { a, b, c } } }
    }

    pub fn e_mul_ev(a: SymbolicExprEF, b: SymbolicExprEF, c: SymbolicVarEF) -> Self {
        Self { opcode: Opcode::EMulEV, args: Arguments { e_op_ev: EOperationEV { a, b, c } } }
    }

    pub fn e_mul_ee(a: SymbolicExprEF, b: SymbolicExprEF, c: SymbolicExprEF) -> Self {
        Self { opcode: Opcode::EMulEE, args: Arguments { e_op_ee: EOperationEE { a, b, c } } }
    }

    pub fn e_mul_assign_e(a: SymbolicExprEF, b: SymbolicExprEF) -> Self {
        Self { opcode: Opcode::EMulAssignE, args: Arguments { e_op_e: EOperationE { a, b } } }
    }
}

impl Default for Instruction {
    fn default() -> Self {
        Self { opcode: Opcode::Empty, args: Arguments { empty: () } }
    }
}
