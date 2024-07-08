#pragma once

#include "../../fields/bb31_t.cuh"
#include "../../fields/bn254_t.cuh"

// TODO: modify mont_t to take a pointer to a constant array as even instead of copying one into it
// so that poseidon constants can be knwon compile time.

namespace poseidon2_bn254_3 {

namespace constants {

    constexpr const int DIGEST_WIDTH = 1;
    constexpr const int RATE = 2;
    constexpr const int WIDTH = 3;
    constexpr const int ROUNDS_P = 56;
    constexpr const int ROUNDS_F = 8;
    constexpr const int D = 5;

}  // namespace constants

class Bn254 {
  public:
    using F_t = bn254_t;
    using pF_t = F_t;

    static constexpr const int DIGEST_WIDTH = constants::DIGEST_WIDTH;
    static constexpr const int RATE = constants::RATE;
    static constexpr const int WIDTH = constants::WIDTH;
    static constexpr const int ROUNDS_F = constants::ROUNDS_F;
    static constexpr const int ROUNDS_P = constants::ROUNDS_P;
    static constexpr const int D = constants::D;

    __device__ static void
    internalLinearLayer(F_t state[WIDTH], F_t matInternalDiagM1[WIDTH], F_t _) {
        F_t s = state[0] + state[1] + state[2];
        for (int i = 0; i < WIDTH; i++) {
            state[i] *= matInternalDiagM1[i];
            state[i] += s;
        }
    }

    __device__ static void externalLinearLayer(F_t state[WIDTH]) {
        F_t sum = state[0] + state[1] + state[2];
        state[0] += sum;
        state[1] += sum;
        state[2] += sum;
    }
};

static __device__ __constant__ __align__(16
) const uint32_t ALT_BN128_1ls32[8] = {
    /* (1<<32) % P_bn254 */
    0x15b8b9da,
    0x93e78865,
    0xb05ea154,
    0x16df2426,
    0x302ab839,
    0x1271b743,
    0xec6c226e,
    0x06bc037e
};

template<int SIZE>
__device__ void
mul_u32_p(uint32_t v, const uint32_t p[SIZE], uint32_t result[SIZE + 1]) {
    uint64_t carry = 0;
    for (int ii = 0; ii < SIZE; ++ii) {
        uint64_t temp = (uint64_t)(v)*p[ii] + carry;
        result[ii] = temp & 0xFFFFFFFF;
        carry = temp >> 32;
    }
    result[SIZE] = (uint32_t)carry;
}

template<int SIZE>
__device__ void substract(uint32_t* a, const uint32_t* b) {
    uint64_t borrow = 0;
    for (int ii = 0; ii < SIZE; ++ii) {
        uint64_t sub = (uint64_t)a[ii] - b[ii] - borrow;
        a[ii] = (uint32_t)(sub & 0xFFFFFFFF);
        borrow = (sub >> 32) & 1;
    }
}

template<int SIZE>
__device__ bool greaterThan(const uint32_t* a, uint32_t* b) {
    for (int i = SIZE - 1; i >= 0; --i) {
        if (a[i] > b[i]) {
            return true;
        }
        if (a[i] < b[i]) {
            return false;
        }
    }
    return false;
}

// TODO: cleaner! size_t vs int, camel case vs snake case

template<int SIZE>
__device__ void modulo_p(
    uint32_t v[SIZE + 1],
    const uint32_t p[SIZE],
    uint32_t left,
    uint32_t rigth
) {
    uint32_t prod[SIZE + 1] = {0};
    while (left <= rigth) {
        uint32_t mid = left + (rigth - left) / 2;
        mul_u32_p<SIZE>(mid, p, prod);
        if (greaterThan<SIZE + 1>(prod, v)) {
            rigth = mid - 1;
        } else {
            if (left == mid)
                break;
            left = mid;
        }
    }
    substract<SIZE + 1>(v, prod);
}

const size_t N = 8;

__device__ bn254_t bb31_to_bn254(bb31_t in) {
    uint32_t canonical = (0x38400000ULL * in.val) % (uint64_t)bb31_t::MOD;
    uint32_t v[N + 1] = {0};
    mul_u32_p<N>(canonical, device::ALT_BN128_rone, v);
    // NOTE: likely not the optimal modulo implementation
    // Max 31 large (8 uint32) multiplications
    modulo_p<N>(v, device::ALT_BN128_r, 0x00000000u, 0x78000001u);
    return bn254_t(v);
}

__device__ bn254_t reduceBabyBear(
    bb31_t* src1,
    bb31_t* src2,
    size_t n1,
    size_t n2,
    size_t stride1 = 1,
    size_t stride2 = 1
) {
    const bn254_t po2 = bn254_t(ALT_BN128_1ls32);
    bn254_t res;
    res.zero();

    if (n2 > 0) {
        for (size_t ii = (n2 - 1) * stride2; true; ii -= stride2) {
            res = res * po2 + bb31_to_bn254(src2[ii]);
            if (ii < stride2)
                break;
        }
    }
    if (n1 > 0) {
        for (size_t ii = (n1 - 1) * stride1; true; ii -= stride1) {
            res = res * po2 + bb31_to_bn254(src1[ii]);
            if (ii < stride1)
                break;
        }
    }

    return res;
}

template<typename Hasher, typename HasherState>
__device__ void
absorbRow(Hasher hasher, Matrix<bb31_t>* in, int rowIdx, HasherState* state) {
    bb31_t* rowPtr;
    size_t stride;
    if (in->row_major) {
        rowPtr = &in->values[rowIdx * in->width];
        stride = 1;
    } else {
        rowPtr = &in->values[rowIdx];
        stride = in->height;
    }

    int colIdx = 0;

    if (state->overhangSize > 0) {
        if (state->overhangSize + in->width < N) {
            // Overhang + row is smaller than N, copy row into overhang and return
            for (size_t i = 0; i < in->width; i++) {
                state->overhang[state->overhangSize + i] = rowPtr[i * stride];
            }
            state->overhangSize += in->width;
            return;
        } else {
            // Overhang + row is larger or equal to N, create bn254_t value from overhang and row and continue
            colIdx = N - state->overhangSize;
            bn254_t value = reduceBabyBear(
                state->overhang,
                rowPtr,
                state->overhangSize,
                colIdx,
                1,
                stride
            );
            state->overhangSize = 0;
            (*state).absorb(&value, 1, hasher);
        }
    }

    while (colIdx + N <= in->width) {
        bn254_t value =
            reduceBabyBear(rowPtr + colIdx * stride, nullptr, N, 0, stride, 0);
        (*state).absorb(&value, 1, hasher);
        colIdx += N;
    }

    if (colIdx < in->width) {
        // Copy remaining elements into overhang
        for (size_t i = 0; i < in->width - colIdx; i++) {
            state->overhang[state->overhangSize + i] =
                rowPtr[(colIdx + i) * stride];
        }
        state->overhangSize = in->width - colIdx;
    }
}

}  // namespace poseidon2_bn254_3
