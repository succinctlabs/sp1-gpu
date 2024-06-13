
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class Bls12381DoubleAssignAir {
   public:
    static constexpr size_t PREP_WIDTH = 0;
    static constexpr size_t MAIN_WIDTH = 3442;
    static constexpr size_t PERM_WIDTH = 831;
    static constexpr size_t EXPRESSION_COUNT = 526;
};