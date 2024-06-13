
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class ShiftRightAir {
   public:
    static constexpr size_t PREP_WIDTH = 0;
    static constexpr size_t MAIN_WIDTH = 71;
    static constexpr size_t PERM_WIDTH = 14;
    static constexpr size_t EXPRESSION_COUNT = 12;
};