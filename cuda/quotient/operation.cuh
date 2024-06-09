#pragma once

#include <cstdint>
#include "./symbolic_folder_expr.cuh"
#include "./symbolic_folder_var.cuh"

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