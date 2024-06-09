
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class KeccakPermuteAir {
   public:
    static constexpr size_t PREP_WIDTH = 0;
    static constexpr size_t MAIN_WIDTH = 3391;
    static constexpr size_t PERM_WIDTH = 152;
    static constexpr size_t EXPRESSION_COUNT = 61896;
};