
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class Bls12381DecompressAir {
   public:
    static constexpr size_t PREP_WIDTH = 0;
    static constexpr size_t MAIN_WIDTH = 1790;
    static constexpr size_t PERM_WIDTH = 418;
    static constexpr size_t EXPRESSION_COUNT = 478;
};