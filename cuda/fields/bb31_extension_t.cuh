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

    __device__ __forceinline__ bb31_extension_t exp_power_of_two(size_t log_power) {
        bb31_extension_t ret = *this;
        for (size_t i = 0; i < log_power; i++) {
            ret *= ret;
        }
        return ret;
    }

    friend __device__ __forceinline__ bool operator!=(
        const bb31_extension_t& lhs, 
        const bb31_extension_t& rhs
    ) {
        for (int i = 0; i < D; ++i) {
            if (lhs.value[i].val != rhs.value[i].val)
                return true;
        }
        return false;
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
/*
__device__ bb31_extension_t atomicCAS(bb31_extension_t* address, bb31_extension_t compare, bb31_extension_t val) {
    bb31_extension_t old = *address;
    for (int i = 0; i < bb31_extension_t::D; ++i) {
        old.value[i].val = atomicCAS(
            reinterpret_cast<uint32_t*>(&address->value[i].val), 
            compare.value[i].val, 
            val.value[i].val);
    }
    return old;
}

__device__ bb31_extension_t atomicAdd(bb31_extension_t* address, bb31_extension_t value) {
    bb31_extension_t old = *address;
    bb31_extension_t assumed;
    
    do {
        assumed = old;
        bb31_extension_t newVal = assumed + value;
        old = atomicCAS(address, assumed, newVal);
    } while (assumed != old);

    return old;
}
*/
/*
// Atomic Compare-and-Swap (CAS) operation
__device__ bb31_extension_t atomicCAS(bb31_extension_t* address, bb31_extension_t compare, bb31_extension_t new_value) {
    bb31_extension_t old_value = *address;
    if (old_value.value[0] == compare.value[0] &&
        old_value.value[1] == compare.value[1] &&
        old_value.value[2] == compare.value[2] &&
        old_value.value[3] == compare.value[3]) {
        if (atomicCAS(&address->value[0].val, compare.value[0].val, new_value.value[0].val) == compare.value[0].val &&
            atomicCAS(&address->value[1].val, compare.value[1].val, new_value.value[1].val) == compare.value[1].val &&
            atomicCAS(&address->value[2].val, compare.value[2].val, new_value.value[2].val) == compare.value[2].val &&
            atomicCAS(&address->value[3].val, compare.value[3].val, new_value.value[3].val) == compare.value[3].val) {
            return old_value;
        }
    }
    return *address;
}

// Atomic Add operation
__device__ bb31_extension_t atomicAdd(bb31_extension_t* address, bb31_extension_t value) {
    bb31_extension_t old_value, new_value;
    do {
        old_value = *address;
        new_value = old_value + value;
    } while (atomicCAS(address, old_value, new_value) != old_value);
    return new_value;
}
*/
__device__ bb31_t atomicAdd(bb31_t* address, bb31_t value) {
    bb31_t old_val, new_val;
    do {
        old_val = *address;
        new_val = old_val + value;
    } while (atomicCAS((uint32_t*)&(address->val), old_val.val, new_val.val) != old_val.val);
    return new_val;
}

__device__ bb31_extension_t atomicAdd(bb31_extension_t* address, bb31_extension_t value) {
    bb31_extension_t old = *address;
    for (int i = 0; i < bb31_extension_t::D; ++i) {
        old.value[i] = atomicAdd(&address->value[i], value.value[i]);
    }
    return old; 
}