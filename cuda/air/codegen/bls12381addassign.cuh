
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class Bls12381AddAssignAir {
   public:
    static constexpr size_t PREP_WIDTH = 0;
    static constexpr size_t MAIN_WIDTH = 3091;
    static constexpr size_t PERM_WIDTH = 737;
    static constexpr size_t EXPRESSION_COUNT = 82203;
};