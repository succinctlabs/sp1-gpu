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

    FNegE = 25,

    EAssignC = 26,
    EAssignV = 27,
    EAssignE = 28,

    EAddVC = 29,
    EAddVV = 30,
    EAddVE = 31,

    EAddEC = 32,
    EAddEV = 33,
    EAddEE = 34,
    EAddAssignE = 35,

    ESubVC = 36,
    ESubVV = 37,
    ESubVE = 38,

    ESubEC = 39,
    ESubEV = 40,
    ESubEE = 41,
    ESubAssignE = 42,

    EMulVC = 43,
    EMulVV = 44,
    EMulVE = 45,

    EMulEC = 46,
    EMulEV = 47,
    EMulEE = 48,
    EMulAssignE = 49,

    ENegE = 50,

    EFFromE = 51,
    EFAddEE = 52,
    EFAddAssignE = 53,
    EFSubEE = 54,
    EFSubAssignE = 55,
    EFMulEE = 56,
    EFMulAssignE = 57,
    EFAsBaseSlice = 58,

    FAssertZero = 59,
    EAssertZero = 60,
}

#[derive(Clone, Copy)]
#[repr(C)]
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

    pub ef_op_e: EFOperationE,
    pub ef_op_ee: EFOperationEE,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FOperationC {
    pub a: SymbolicExprF,
    pub b: F,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FOperationV {
    pub a: SymbolicExprF,
    pub b: SymbolicVarF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FOperationE {
    pub a: SymbolicExprF,
    pub b: SymbolicExprF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FOperationVC {
    pub a: SymbolicExprF,
    pub b: SymbolicVarF,
    pub c: F,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FOperationVV {
    pub a: SymbolicExprF,
    pub b: SymbolicVarF,
    pub c: SymbolicVarF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FOperationVE {
    pub a: SymbolicExprF,
    pub b: SymbolicVarF,
    pub c: SymbolicExprF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FOperationEC {
    pub a: SymbolicExprF,
    pub b: SymbolicExprF,
    pub c: F,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FOperationEV {
    pub a: SymbolicExprF,
    pub b: SymbolicExprF,
    pub c: SymbolicVarF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct FOperationEE {
    pub a: SymbolicExprF,
    pub b: SymbolicExprF,
    pub c: SymbolicExprF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct EOperationC {
    pub a: SymbolicExprEF,
    pub b: EF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct EOperationV {
    pub a: SymbolicExprEF,
    pub b: SymbolicVarEF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct EOperationE {
    pub a: SymbolicExprEF,
    pub b: SymbolicExprEF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct EOperationVC {
    pub a: SymbolicExprEF,
    pub b: SymbolicVarEF,
    pub c: EF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct EOperationVV {
    pub a: SymbolicExprEF,
    pub b: SymbolicVarEF,
    pub c: SymbolicVarEF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct EOperationVE {
    pub a: SymbolicExprEF,
    pub b: SymbolicVarEF,
    pub c: SymbolicExprEF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct EOperationEC {
    pub a: SymbolicExprEF,
    pub b: SymbolicExprEF,
    pub c: EF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct EOperationEV {
    pub a: SymbolicExprEF,
    pub b: SymbolicExprEF,
    pub c: SymbolicVarEF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct EOperationEE {
    pub a: SymbolicExprEF,
    pub b: SymbolicExprEF,
    pub c: SymbolicExprEF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct EFOperationE {
    pub a: SymbolicExprEF,
    pub b: SymbolicExprF,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct EFOperationEE {
    pub a: SymbolicExprEF,
    pub b: SymbolicExprEF,
    pub c: SymbolicExprF,
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

    pub fn f_neg_e(a: SymbolicExprF, b: SymbolicExprF) -> Self {
        Self { opcode: Opcode::FNegE, args: Arguments { f_op_e: FOperationE { a, b } } }
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

    pub fn e_neg_e(a: SymbolicExprEF, b: SymbolicExprEF) -> Self {
        Self { opcode: Opcode::ENegE, args: Arguments { e_op_e: EOperationE { a, b } } }
    }

    pub fn ef_from_e(a: SymbolicExprEF, b: SymbolicExprF) -> Self {
        Self { opcode: Opcode::EFFromE, args: Arguments { ef_op_e: EFOperationE { a, b } } }
    }

    pub fn ef_add_ee(a: SymbolicExprEF, b: SymbolicExprEF, c: SymbolicExprF) -> Self {
        Self { opcode: Opcode::EFAddEE, args: Arguments { ef_op_ee: EFOperationEE { a, b, c } } }
    }

    pub fn ef_add_assign_e(a: SymbolicExprEF, b: SymbolicExprF) -> Self {
        Self { opcode: Opcode::EFAddAssignE, args: Arguments { ef_op_e: EFOperationE { a, b } } }
    }

    pub fn ef_sub_ee(a: SymbolicExprEF, b: SymbolicExprEF, c: SymbolicExprF) -> Self {
        Self { opcode: Opcode::EFSubEE, args: Arguments { ef_op_ee: EFOperationEE { a, b, c } } }
    }

    pub fn ef_sub_assign_e(a: SymbolicExprEF, b: SymbolicExprF) -> Self {
        Self { opcode: Opcode::EFSubAssignE, args: Arguments { ef_op_e: EFOperationE { a, b } } }
    }

    pub fn ef_mul_ee(a: SymbolicExprEF, b: SymbolicExprEF, c: SymbolicExprF) -> Self {
        Self { opcode: Opcode::EFMulEE, args: Arguments { ef_op_ee: EFOperationEE { a, b, c } } }
    }

    pub fn ef_mul_assign_e(a: SymbolicExprEF, b: SymbolicExprF) -> Self {
        Self { opcode: Opcode::EFMulAssignE, args: Arguments { ef_op_e: EFOperationE { a, b } } }
    }

    pub fn f_assert_zero(a: SymbolicExprF) -> Self {
        Self {
            opcode: Opcode::FAssertZero,
            args: Arguments { f_op_e: FOperationE { a, b: SymbolicExprF::alloc() } },
        }
    }

    pub fn e_assert_zero(a: SymbolicExprEF) -> Self {
        Self {
            opcode: Opcode::EAssertZero,
            args: Arguments { e_op_e: EOperationE { a, b: SymbolicExprEF::alloc() } },
        }
    }
}

impl Default for Instruction {
    fn default() -> Self {
        Self { opcode: Opcode::Empty, args: Arguments { empty: () } }
    }
}

impl Debug for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            match self.opcode {
                Opcode::Empty => write!(f, "Empty"),
                Opcode::FAssignC => write!(f, "FAssignC({:?})", self.args.f_op_c),
                Opcode::FAssignV => write!(f, "FAssignV({:?})", self.args.f_op_v),
                Opcode::FAssignE => write!(f, "FAssignE({:?})", self.args.f_op_e),

                Opcode::FAddVC => write!(f, "FAddVC({:?})", self.args.f_op_vc),
                Opcode::FAddVV => write!(f, "FAddVV({:?})", self.args.f_op_vv),
                Opcode::FAddVE => write!(f, "FAddVE({:?})", self.args.f_op_ve),

                Opcode::FAddEC => write!(f, "FAddEC({:?})", self.args.f_op_ec),
                Opcode::FAddEV => write!(f, "FAddEV({:?})", self.args.f_op_ev),
                Opcode::FAddEE => write!(f, "FAddEE({:?})", self.args.f_op_ee),
                Opcode::FAddAssignE => write!(f, "FAddAssignE({:?})", self.args.f_op_e),

                Opcode::FSubVC => write!(f, "FSubVC({:?})", self.args.f_op_vc),
                Opcode::FSubVV => write!(f, "FSubVV({:?})", self.args.f_op_vv),
                Opcode::FSubVE => write!(f, "FSubVE({:?})", self.args.f_op_ve),

                Opcode::FSubEC => write!(f, "FSubEC({:?})", self.args.f_op_ec),
                Opcode::FSubEV => write!(f, "FSubEV({:?})", self.args.f_op_ev),
                Opcode::FSubEE => write!(f, "FSubEE({:?})", self.args.f_op_ee),
                Opcode::FSubAssignE => write!(f, "FSubAssignE({:?})", self.args.f_op_e),

                Opcode::FMulVC => write!(f, "FMulVC({:?})", self.args.f_op_vc),
                Opcode::FMulVV => write!(f, "FMulVV({:?})", self.args.f_op_vv),
                Opcode::FMulVE => write!(f, "FMulVE({:?})", self.args.f_op_ve),

                Opcode::FMulEC => write!(f, "FMulEC({:?})", self.args.f_op_ec),
                Opcode::FMulEV => write!(f, "FMulEV({:?})", self.args.f_op_ev),
                Opcode::FMulEE => write!(f, "FMulEE({:?})", self.args.f_op_ee),
                Opcode::FMulAssignE => write!(f, "FMulAssignE({:?})", self.args.f_op_e),

                Opcode::FNegE => write!(f, "FNegE({:?})", self.args.f_op_e),

                Opcode::EAssignC => write!(f, "EAssignC({:?})", self.args.e_op_c),
                Opcode::EAssignV => write!(f, "EAssignV({:?})", self.args.e_op_v),
                Opcode::EAssignE => write!(f, "EAssignE({:?})", self.args.e_op_e),

                Opcode::EAddVC => write!(f, "EAddVC({:?})", self.args.e_op_vc),
                Opcode::EAddVV => write!(f, "EAddVV({:?})", self.args.e_op_vv),
                Opcode::EAddVE => write!(f, "EAddVE({:?})", self.args.e_op_ve),

                Opcode::EAddEC => write!(f, "EAddEC({:?})", self.args.e_op_ec),
                Opcode::EAddEV => write!(f, "EAddEV({:?})", self.args.e_op_ev),
                Opcode::EAddEE => write!(f, "EAddEE({:?})", self.args.e_op_ee),
                Opcode::EAddAssignE => write!(f, "EAddAssignE({:?})", self.args.e_op_e),

                Opcode::ESubVC => write!(f, "ESubVC({:?})", self.args.e_op_vc),
                Opcode::ESubVV => write!(f, "ESubVV({:?})", self.args.e_op_vv),
                Opcode::ESubVE => write!(f, "ESubVE({:?})", self.args.e_op_ve),

                Opcode::ESubEC => write!(f, "ESubEC({:?})", self.args.e_op_ec),
                Opcode::ESubEV => write!(f, "ESubEV({:?})", self.args.e_op_ev),
                Opcode::ESubEE => write!(f, "ESubEE({:?})", self.args.e_op_ee),
                Opcode::ESubAssignE => write!(f, "ESubAssignE({:?})", self.args.e_op_e),

                Opcode::EMulVC => write!(f, "EMulVC({:?})", self.args.e_op_vc),
                Opcode::EMulVV => write!(f, "EMulVV({:?})", self.args.e_op_vv),
                Opcode::EMulVE => write!(f, "EMulVE({:?})", self.args.e_op_ve),

                Opcode::EMulEC => write!(f, "EMulEC({:?})", self.args.e_op_ec),
                Opcode::EMulEV => write!(f, "EMulEV({:?})", self.args.e_op_ev),
                Opcode::EMulEE => write!(f, "EMulEE({:?})", self.args.e_op_ee),
                Opcode::EMulAssignE => write!(f, "EMulAssignE({:?})", self.args.e_op_e),

                Opcode::ENegE => write!(f, "ENegE({:?})", self.args.e_op_e),

                Opcode::EFFromE => {
                    write!(f, "EFFromE({:?})", self.args.ef_op_e)
                }
                Opcode::EFAddEE => write!(f, "EFAddEE({:?})", self.args.ef_op_ee),
                Opcode::EFAddAssignE => write!(f, "EFAddAssignE({:?})", self.args.ef_op_e),
                Opcode::EFSubEE => write!(f, "EFSubEE({:?})", self.args.ef_op_ee),
                Opcode::EFSubAssignE => write!(f, "EFSubAssignE({:?})", self.args.ef_op_e),
                Opcode::EFMulEE => write!(f, "EFMulEE({:?})", self.args.ef_op_ee),
                Opcode::EFMulAssignE => write!(f, "EFMulAssignE({:?})", self.args.ef_op_e),
                Opcode::EFAsBaseSlice => write!(f, "EFAsBaseSlice({:?})", self.args.ef_op_e),

                Opcode::FAssertZero => write!(f, "FAssertZero({:?})", self.args.f_op_e),
                Opcode::EAssertZero => write!(f, "EAssertZero({:?})", self.args.e_op_e),
            }
        }
    }
}
