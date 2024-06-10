
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class Secp256k1DoubleAssignAir {
   public:
    static constexpr size_t PREP_WIDTH = 0;
    static constexpr size_t MAIN_WIDTH = 2282;
    static constexpr size_t PERM_WIDTH = 551;
    static constexpr size_t EXPRESSION_COUNT = 54963;
};