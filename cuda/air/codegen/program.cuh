
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class ProgramAir {
   public:
    __device__ void eval(ConstraintFolder<bb31_t, bb31_extension_t, 2> *builder) {
       bb31_extension_t tmp0 = bb31_t{0} + builder->permLocal[0];
       bb31_extension_t tmp1 = bb31_t{0} + builder->permNext[0];
       bb31_extension_t tmp2 = builder->permNext[1] - builder->permLocal[1];
       bb31_extension_t tmp3 = tmp2 - tmp1;
       bb31_extension_t tmp4 = tmp3 * builder->isTransition;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp4;
       bb31_extension_t tmp5 = builder->permLocal[1] - tmp0;
       bb31_extension_t tmp6 = tmp5 * builder->isFirstRow;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp6;
       bb31_extension_t tmp7 = builder->permLocal[1] - builder->mainLocal[1000];
       bb31_extension_t tmp8 = tmp7 * builder->isLastRow;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp8;
    }
};