#pragma once

#include <cstdint>
#include "../fields/bb31_extension_t.cuh"

enum SymbolicVarFKind : size_t {
    EmptyF = 0,
    ConstantF = 1,
    PreprocessedLocal = 2,
    PreprocessedNext = 3,
    MainLocal = 4,
    MainNext = 5,
    IsFirstRow = 6,
    IsLastRow = 7,
    IsTransition = 8,
    PublicValue = 9,
};

union SymbolicVarFArgs {
    bb31_t f; 
    uint16_t idx;
};

struct SymbolicVarF {
    SymbolicVarFKind variant;
    SymbolicVarFArgs args;
};

enum SymbolicVarEFKind {
    EmptyEF = 0,
    ConstantEF = 1,
    PermutationLocal = 2,
    PermutationNext = 3,
    PermutationChallenge = 4,
    CumulativeSum = 5,
};  

union SymbolicVarEFArgs {
    bb31_t ef;
    uint16_t idx;
};

struct SymbolicVarEF {
    SymbolicVarEFKind variant;
    SymbolicVarEFArgs args;
};

struct SymbolicExprF {
    uint32_t id;
};

struct SymbolicExprEF {
    uint32_t id;
};

enum Opcode : size_t {
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
};

struct FOperationC {
    SymbolicExprF a;
    bb31_t b;
};

struct FOperationV {
    SymbolicExprF a;
    SymbolicVarF b;
};

struct FOperationE {
    SymbolicExprF a;
    SymbolicExprF b;
};

struct FOperationVC {
    SymbolicExprF a;
    SymbolicVarF b;
    bb31_t c;
};

struct FOperationVV {
    SymbolicExprF a;
    SymbolicVarF b;
    SymbolicVarF c;
};

struct FOperationVE {
    SymbolicExprF a;
    SymbolicVarF b;
    SymbolicExprF c;
};

struct FOperationEC {
    SymbolicExprF a;
    SymbolicExprF b;
    bb31_t c;
};

struct FOperationEV {
    SymbolicExprF a;
    SymbolicExprF b;
    SymbolicVarF c;
};

struct FOperationEE {
    SymbolicExprF a;
    SymbolicExprF b;
    SymbolicExprF c;
};

struct EOperationC {
    SymbolicExprEF a;
    bb31_t b;
};

struct EOperationV {
    SymbolicExprEF a;
    SymbolicVarEF b;
};

struct EOperationE {
    SymbolicExprEF a;
    SymbolicExprEF b;
};

struct EOperationVC {
    SymbolicExprEF a;
    SymbolicVarEF b;
    bb31_t c;
};

struct EOperationVV {
    SymbolicExprEF a;
    SymbolicVarEF b;
    SymbolicVarEF c;
};

struct EOperationVE {
    SymbolicExprEF a;
    SymbolicVarEF b;
    SymbolicExprEF c;
};

struct EOperationEC {
    SymbolicExprEF a;
    SymbolicExprEF b;
    bb31_t c;
};

struct EOperationEV {
    SymbolicExprEF a;
    SymbolicExprEF b;
    SymbolicVarEF c;
};

struct EOperationEE {
    SymbolicExprEF a;
    SymbolicExprEF b;
    SymbolicExprEF c;
};

struct EFOperationE {
    SymbolicExprEF a;
    SymbolicExprF b;
};

struct EFOperationEE {
    SymbolicExprEF a;
    SymbolicExprEF b;
    SymbolicExprF c;
};

union Arguments {
    FOperationC f_op_c;
    FOperationV f_op_v;
    FOperationE f_op_e;

    FOperationVC f_op_vc;
    FOperationVV f_op_vv;
    FOperationVE f_op_ve;

    FOperationEC f_op_ec;
    FOperationEV f_op_ev;
    FOperationEE f_op_ee;

    EOperationC e_op_c;
    EOperationV e_op_v;
    EOperationE e_op_e;

    EOperationVC e_op_vc;
    EOperationVV e_op_vv;
    EOperationVE e_op_ve;

    EOperationEC e_op_ec;
    EOperationEV e_op_ev;
    EOperationEE e_op_ee;

    EFOperationE ef_op_e;
    EFOperationEE ef_op_ee;
};

struct Instruction {
    Opcode opcode;
    Arguments args;
};

template <typename Val>
struct LagrangeSelectorsAtPoint {
    Val is_first_row;
    Val is_last_row;
    Val is_transition;
    Val inv_zeroifier;
};

template <typename Val>
struct LagrangeSelectors {
    Val *is_first_row;
    Val *is_last_row;
    Val *is_transition;
    Val *inv_zeroifier;
};

template <typename Val>
struct TwoAdicMultiplicativeCoset {
    size_t log_n;
    Val shift;

    __device__ __forceinline__ size_t size() const { return 1 << log_n; }

    __device__ __forceinline__ LagrangeSelectorsAtPoint<Val> selectors_at_point(Val gen, Val point) const {
        Val unshifted_point = point / shift;
        Val z_h = unshifted_point.exp_power_of_two(log_n) - Val::one();

        Val gen_inverse = gen.reciprocal();

        Val is_first_row = z_h/ (unshifted_point - Val::one());
        Val is_last_row = z_h / (unshifted_point - gen_inverse);
        Val is_transition = unshifted_point - gen_inverse;
        Val inv_zeroifier = z_h.reciprocal();

        LagrangeSelectorsAtPoint<Val> ret;
        ret.is_first_row = is_first_row;
        ret.is_last_row = is_last_row;
        ret.is_transition = is_transition;
        ret.inv_zeroifier = inv_zeroifier;

        return ret;
    }
};

