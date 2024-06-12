
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class CPUAir {
   public:
    static constexpr size_t PREP_WIDTH = 0;
    static constexpr size_t MAIN_WIDTH = 151;
    static constexpr size_t PERM_WIDTH = 18;
    static constexpr size_t EXPRESSION_COUNT = 60;
};