
#pragma once

#include "../../fields/bb31_extension_t.cuh"
#include "../folder.cuh"

class BitwiseAir {
   public:
    __device__ void eval(ConstraintFolder<bb31_t, bb31_extension_t, 2> *builder) {
       bb31_extension_t tmp0 = builder->isFirstRow * builder->mainLocal[2];
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp0;
       bb31_extension_t tmp1 = bb31_t{1};
       bb31_extension_t tmp2 = builder->mainLocal[2] + tmp1;
       bb31_extension_t tmp3 = tmp2 - builder->mainNext[2];
       bb31_extension_t tmp4 = builder->isTransition * tmp3;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp4;
       bb31_extension_t tmp5 = builder->mainLocal[15] * bb31_t{2};
       bb31_extension_t tmp6 = builder->mainLocal[16] * bb31_t{1};
       bb31_extension_t tmp7 = tmp5 + tmp6;
       bb31_extension_t tmp8 = builder->mainLocal[17] * bb31_t{0};
       bb31_extension_t tmp9 = tmp7 + tmp8;
       bb31_extension_t tmp10 = builder->mainLocal[15] + builder->mainLocal[16];
       bb31_extension_t tmp11 = tmp10 + builder->mainLocal[17];
       bb31_extension_t tmp12 = bb31_t{0};
       bb31_extension_t tmp13 = bb31_t{0};
       bb31_extension_t tmp14 = bb31_t{0};
       bb31_extension_t tmp15 = bb31_t{0};
       bb31_extension_t tmp16 = builder->mainLocal[15] * bb31_t{2};
       bb31_extension_t tmp17 = builder->mainLocal[16] * bb31_t{3};
       bb31_extension_t tmp18 = tmp16 + tmp17;
       bb31_extension_t tmp19 = builder->mainLocal[17] * bb31_t{4};
       bb31_extension_t tmp20 = tmp18 + tmp19;
       bb31_extension_t tmp21 = builder->mainLocal[15] + builder->mainLocal[16];
       bb31_extension_t tmp22 = tmp21 + builder->mainLocal[17];
       bb31_extension_t tmp23 = builder->mainLocal[15] + builder->mainLocal[16];
       bb31_extension_t tmp24 = tmp23 + builder->mainLocal[17];
       bb31_extension_t tmp25 = bb31_t{1};
       bb31_extension_t tmp26 = builder->mainLocal[15] - tmp25;
       bb31_extension_t tmp27 = builder->mainLocal[15] * tmp26;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp27;
       bb31_extension_t tmp28 = bb31_t{1};
       bb31_extension_t tmp29 = builder->mainLocal[16] - tmp28;
       bb31_extension_t tmp30 = builder->mainLocal[16] * tmp29;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp30;
       bb31_extension_t tmp31 = bb31_t{1};
       bb31_extension_t tmp32 = builder->mainLocal[17] - tmp31;
       bb31_extension_t tmp33 = builder->mainLocal[17] * tmp32;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp33;
       bb31_extension_t tmp34 = bb31_t{1};
       bb31_extension_t tmp35 = tmp24 - tmp34;
       bb31_extension_t tmp36 = tmp24 * tmp35;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp36;
       bb31_extension_t tmp37 = bb31_t{0} + builder->permLocal[0];
       bb31_extension_t tmp38 = tmp37 + builder->permLocal[1];
       bb31_extension_t tmp39 = tmp38 + builder->permLocal[2];
       bb31_extension_t tmp40 = bb31_t{0} + builder->permNext[0];
       bb31_extension_t tmp41 = tmp40 + builder->permNext[1];
       bb31_extension_t tmp42 = tmp41 + builder->permNext[2];
       bb31_extension_t tmp43 = builder->permNext[3] - builder->permLocal[3];
       bb31_extension_t tmp44 = tmp43 - tmp42;
       bb31_extension_t tmp45 = tmp44 * builder->isTransition;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp45;
       bb31_extension_t tmp46 = builder->permLocal[3] - tmp39;
       bb31_extension_t tmp47 = tmp46 * builder->isFirstRow;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp47;
       bb31_extension_t tmp48 = builder->permLocal[3] - builder->mainLocal[1000];
       bb31_extension_t tmp49 = tmp48 * builder->isLastRow;
       builder->accumulator *= builder->alpha;
       builder->accumulator += tmp49;
    }
};