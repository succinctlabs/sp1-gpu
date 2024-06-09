
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class ShaCompressAir {
   public:
    static constexpr size_t PREP_WIDTH = 0;
    static constexpr size_t MAIN_WIDTH = 264;
    static constexpr size_t PERM_WIDTH = 59;
    static constexpr size_t EXPRESSION_COUNT = 4873;
};