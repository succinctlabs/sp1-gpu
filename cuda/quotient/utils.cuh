#pragma once

#include <cstdint>
#include "../fields/bb31_extension_t.cuh"

enum class SymbolicFolderVarType : std::uint32_t {
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
};


struct SymbolicFolderVar {
    SymbolicFolderVarType variant;
    bb31_t f;
    bb31_extension_t ef;
    size_t idx;
};

struct SymbolicFolderExpr {
    size_t value;
};

enum class OperationType : std::uint32_t {
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
};

struct Operation {
    OperationType variant;
    SymbolicFolderExpr a;
    bb31_t b_f;
    bb31_extension_t b_ef;
    SymbolicFolderVar b_var;
    SymbolicFolderExpr b_expr;
    bb31_t c_f;
    bb31_extension_t c_ef;
    SymbolicFolderVar c_var;
    SymbolicFolderExpr c_expr;
};

template <typename Val>
struct TwoAdicMultiplicativeCoset {
    size_t log_n;
    Val shift;

    __device__ size_t size() const { return 1 << log_n; }
};

template <typename Val>
struct LagrangeSelectors {
    Val *is_first_row;
    Val *is_last_row;
    Val *is_transition;
    Val *inv_zeroifier;
};