
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class MemoryInitAir {
   public:
    static constexpr size_t PREP_WIDTH = 0;
    static constexpr size_t MAIN_WIDTH = 168;
    static constexpr size_t PERM_WIDTH = 2;
    static constexpr size_t EXPRESSION_COUNT = 8;
};