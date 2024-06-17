
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class Secp256k1AddAssignAir {
   public:
    static constexpr size_t PREP_WIDTH = 0;
    static constexpr size_t MAIN_WIDTH = 2051;
    static constexpr size_t PERM_WIDTH = 489;
    static constexpr size_t EXPRESSION_COUNT = 318;
};