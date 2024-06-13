
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class DivRemAir {
   public:
    static constexpr size_t PREP_WIDTH = 0;
    static constexpr size_t MAIN_WIDTH = 104;
    static constexpr size_t PERM_WIDTH = 10;
    static constexpr size_t EXPRESSION_COUNT = 16;
};