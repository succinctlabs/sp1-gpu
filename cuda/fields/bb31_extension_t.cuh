#pragma once

#include "bb31_t.cuh"

class bb31_extension_t {
   public:
    static constexpr size_t D = 4;
    static constexpr bb31_t W = bb31_t {11};

    bb31_t value[D];

    __device__ __forceinline__ bb31_extension_t() {}

    __device__ __forceinline__ bb31_extension_t(bb31_t value[4]) {
        for (size_t i = 0; i < D; i++) {
            this->value[i] = value[i];
        }
    }

    __device__ __forceinline__ bb31_extension_t(bb31_t value) {
        this->value[0] = value;
        for (size_t i = 1; i < D; i++) {
            this->value[i] = bb31_t(0);
        }
    }

    static __device__ __forceinline__ const bb31_extension_t zero() {
        bb31_t values[D] = {bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0)};
        return bb31_extension_t(values);
    }

    static __device__ __forceinline__ const bb31_extension_t one() {
        bb31_t values[D] = {bb31_t::one(), bb31_t(0), bb31_t(0), bb31_t(0)};
        return bb31_extension_t(values);
    }

    __device__ __forceinline__ bb31_extension_t& operator+=(const bb31_extension_t b) {
        for (size_t i = 0; i < D; i++) {
            value[i] += b.value[i];
        }
        return *this;
    }

    friend __device__ __forceinline__ bb31_extension_t operator+(bb31_extension_t a,
                                             const bb31_extension_t b) {
        return a += b;
    }

    __device__ __forceinline__ bb31_extension_t& operator-=(const bb31_extension_t b) {
        for (size_t i = 0; i < D; i++) {
            value[i] -= b.value[i];
        }
        return *this;
    }

    friend __device__ __forceinline__ bb31_extension_t operator-(bb31_extension_t a,
                                             const bb31_extension_t b) {
        return a -= b;
    }

    __device__ __forceinline__ bb31_extension_t& operator*=(const bb31_extension_t b) {
        bb31_t product[4] = {bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0)};
        for (size_t i = 0; i < D; i++) {
            for (size_t j = 0; j < D; j++) {
                if (i + j >= D) {
                    product[i + j - D] += value[i] * b.value[j] * W;
                } else {
                    product[i + j] += value[i] * b.value[j];
                }
            }
        }

        for (size_t i = 0; i < D; i++) {
            value[i] = product[i];
        }

        return *this;
    }

    __device__ __forceinline__ bb31_extension_t& operator*=(const bb31_t b) {
        for (size_t i = 0; i < D; i++) {
            value[i] *= b;
        }
        return *this;
    }

    friend __device__ __forceinline__ bb31_extension_t operator*(bb31_extension_t a,
                                             const bb31_extension_t b) {
        return a *= b;
    }

    friend __device__ __forceinline__ bb31_extension_t operator*(bb31_extension_t a,
                                             const bb31_t b) {
        return a *= b;
    }

    __device__ __forceinline__ bb31_extension_t& operator/=(const bb31_extension_t b) {
        *this *= b.reciprocal();
        return *this;
    }

    friend __device__ __forceinline__ bb31_extension_t operator/(bb31_extension_t a,
                                             const bb31_extension_t b) {
        return a /= b;
    }

    __device__ __forceinline__ bb31_extension_t frobenius() {
        bb31_t z0 = bb31_t(1728404513);
        bb31_t z = z0;
        bb31_extension_t result;
        for (size_t i = 0; i < D; i++) {
            result.value[i] = value[i] * z;
            z *= z0;
        }
        return result;
    }

    __device__ __forceinline__ bb31_extension_t frobeniusInverse() const {
        bb31_extension_t f = one();
        for (size_t i = 1; i < D; i++) {
            f = (f * *this).frobenius();
        }

        bb31_extension_t a = *this;
        bb31_extension_t b = f;
        bb31_t g = bb31_t(0);
        for (size_t i = 1; i < D; i++) {
            g += a.value[i] * b.value[4 - i];
        }
        g *= bb31_t(11);
        g += a.value[0] * b.value[0];
        return f * g.reciprocal();
    }

    __device__ __forceinline__ bb31_extension_t reciprocal() const {
        bool isZero = true;
        for (size_t i = 0; i < D; i++) {
            if (value[i].val != 0) {
                isZero = false;
                break;
            }
        }

        if (isZero) {
            return zero();
        }

        return frobeniusInverse();
    }
};