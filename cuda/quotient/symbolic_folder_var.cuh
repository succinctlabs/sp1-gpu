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