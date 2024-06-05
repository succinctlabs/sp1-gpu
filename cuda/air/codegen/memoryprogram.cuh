
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class MemoryProgramAir {
   public:
    __device__ void eval(ConstraintFolder<bb31_t, bb31_extension_t, 2> *builder) {
       bb31_extension_t tmp0 = bb31_t{1};
       bb31_extension_t tmp1 = builder->mainLocal[0] - tmp0;
       bb31_extension_t tmp2 = builder->mainLocal[0] * tmp1;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp2;
       bb31_extension_t tmp3 = builder->mainLocal[0] - builder->prepLocal[5];
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp3;
       bb31_extension_t tmp4 = bb31_t{1};
       bb31_extension_t tmp5 = builder->publicValues[43] - tmp4;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp5;
       bb31_extension_t tmp6 = bb31_t{0};
       bb31_extension_t tmp7 = bb31_t{0};
       bb31_extension_t tmp8 = bb31_t{0} + builder->permLocal[0];
       bb31_extension_t tmp9 = bb31_t{0} + builder->permNext[0];
       bb31_extension_t tmp10 = builder->permNext[1] - builder->permLocal[1];
       bb31_extension_t tmp11 = tmp10 - tmp9;
       bb31_extension_t tmp12 = tmp11 * builder->isTransition;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp12;
       bb31_extension_t tmp13 = builder->permLocal[1] - tmp8;
       bb31_extension_t tmp14 = tmp13 * builder->isFirstRow;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp14;
       bb31_extension_t tmp15 = builder->permLocal[1] - builder->mainLocal[1000];
       bb31_extension_t tmp16 = tmp15 * builder->isLastRow;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp16;
    }
};