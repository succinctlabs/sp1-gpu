#pragma once

#include <cstdint>
#include "../fields/bb31_extension_t.cuh"

struct Instruction {
    unsigned char opcode;
    unsigned char b_variant;
    unsigned char c_variant;
    unsigned int a;
    unsigned int b;
    unsigned int c;
};

template <typename Val>
struct LagrangeSelectorsAtPoint {
    Val is_first_row;
    Val is_last_row;
    Val is_transition;
    Val inv_zeroifier;
};

template <typename Val>
struct LagrangeSelectors {
    Val *is_first_row;
    Val *is_last_row;
    Val *is_transition;
    Val *inv_zeroifier;
};

template <typename Val>
struct TwoAdicMultiplicativeCoset {
    size_t log_n;
    Val shift;

    __device__ __forceinline__ size_t size() const { return 1 << log_n; }

    __device__ __forceinline__ LagrangeSelectorsAtPoint<Val> selectors_at_point(Val gen, Val point) const {
        Val unshifted_point = point / shift;
        Val z_h = unshifted_point.exp_power_of_two(log_n) - Val::one();

        Val gen_inverse = gen.reciprocal();

        Val is_first_row = z_h/ (unshifted_point - Val::one());
        Val is_last_row = z_h / (unshifted_point - gen_inverse);
        Val is_transition = unshifted_point - gen_inverse;
        Val inv_zeroifier = z_h.reciprocal();

        LagrangeSelectorsAtPoint<Val> ret;
        ret.is_first_row = is_first_row;
        ret.is_last_row = is_last_row;
        ret.is_transition = is_transition;
        ret.inv_zeroifier = inv_zeroifier;

        return ret;
    }
};

