#pragma once

#include "bb31_t.cuh"

class bb31_septic_extension_t {
   public:
    static constexpr size_t D = 7;

    bb31_t value[D];

    __device__ __forceinline__ bb31_septic_extension_t() {}

    __device__ __forceinline__ bb31_septic_extension_t(bb31_t value[7]) {
        for (size_t i = 0; i < D; i++) {
            this->value[i] = value[i];
        }
    }

    __device__ __forceinline__ bb31_septic_extension_t(bb31_t value) {
        this->value[0] = value;
        for (size_t i = 1; i < D; i++) {
            this->value[i] = bb31_t(0);
        }
    }
};

class bb31_septic_curve_t {
   public:
    static constexpr size_t D = 7;
    bb31_septic_extension_t x;
    bb31_septic_extension_t y;

    __device__ __forceinline__ bb31_septic_curve_t() {}

    __device__ __forceinline__ bb31_septic_curve_t(bb31_t value[14]) {
        for (size_t i = 0; i < D; i++) {
            this->x.value[i] = value[i];
        }
        for (size_t i = 0; i < D; i++) {
            this->y.value[i] = value[i+D];
        }
    }

    __device__ __forceinline__ bb31_septic_curve_t(bb31_septic_extension_t x, bb31_septic_extension_t y) {
        this->x = x;
        this->y = y;
    }
};

class bb31_septic_digest_t {
   public:
    bb31_septic_curve_t point;

    __device__ __forceinline__ bb31_septic_digest_t() {}

    __device__ __forceinline__ bb31_septic_digest_t(bb31_t value[14]) {
        this->point = bb31_septic_curve_t(value);
    }

    __device__ __forceinline__ bb31_septic_digest_t(bb31_septic_extension_t x, bb31_septic_extension_t y) {
        this->point = bb31_septic_curve_t(x, y);
    }

    __device__ __forceinline__ bb31_septic_digest_t(bb31_septic_curve_t point) {
        this->point = point;
    }
};