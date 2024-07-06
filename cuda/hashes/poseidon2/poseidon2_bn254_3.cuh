#pragma once

#include "../../fields/bb31_t.cuh"
#include "../../fields/bn254_t.cuh"

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

// TODO: rename, clean, organize

// bn254_t::nbits / bb31_t::nbits
const size_t ARRAY_SIZE = 8;

static __device__ __constant__ __align__(16
) const uint32_t ALT_BN128_1ls32[ARRAY_SIZE] = {
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

static __device__ __constant__ __align__(16
) const uint32_t ALT_BN128_r[ARRAY_SIZE + 1] = {
    /* (1<<256)%P_bn254 */
    0xf0000001,
    0x43e1f593,
    0x79b97091,
    0x2833e848,
    0x8181585d,
    0xb85045b6,
    0xe131a029,
    0x30644e72,
    0x00000000,
};

__device__ void printArray(uint32_t* arr, size_t size) {
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadId != 0) {
        return;
    }
    for (size_t i = 0; i < size; i++) {
        printf("%lu ", arr[i]);
    }
    printf("\n");
}

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
__device__ bool greater_than(const uint32_t* a, uint32_t* b) {
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

__device__ bn254_t bb31_to_bn254(bb31_t in) {
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadId == 0) {
        printf("GPU: reduceBabyBear %lu\n", in.val);
    }

    if (threadId == 0) printf("1");

    uint32_t canonical = (0x38400000ULL * in.val) % (uint64_t)bb31_t::MOD;
    uint32_t product[ARRAY_SIZE + 1] = {0};
    mul_u32_p<ARRAY_SIZE>(canonical, device::ALT_BN128_rone, product);

    if (threadId == 0)
        printf("1");

    if (threadId == 0) {
        printf("product: ");
        printArray(product, ARRAY_SIZE + 1);
    }

    if (threadId == 0)
        printf("1");

    uint32_t qLeft = 0;
    uint32_t qRight = 1 << 31;

    uint32_t p[ARRAY_SIZE + 1] = {0};

    int loops = 0;

    if (threadId == 0)
        printf("1");

    // At most 32 big multiplications
    while (qLeft <= qRight) {
        loops++;
        uint32_t qMiddle = qLeft + (qRight - qLeft) / 2;
        mul_u32_p<ARRAY_SIZE>(qMiddle, device::ALT_BN128_r, p);
        if (greater_than<ARRAY_SIZE + 1>(p, product)) {
            qRight = qMiddle - 1;
        } else {
            if (qLeft == qMiddle)
                break;
            qLeft = qMiddle;
        }
    }

    if (threadId == 0)
        printf("1");

    if (threadId == 0) {
        printf("loops %d\n", loops);
        printf("quotient: %lu\n", qLeft);
        printf("quotient: ");
        printf("%lu\n", qLeft);
    }

    if (threadId == 0)
        printf("1");

    substract<ARRAY_SIZE + 1>(product, p);
    uint32_t mod[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        mod[i] = product[i];
    }

    // printf("mod: ");
    // printArray(mod, ARRAY_SIZE);

    return bn254_t(mod);
}

__device__ bn254_t reduceBabyBear(
    bb31_t* src1,
    bb31_t* src2,
    size_t n1,
    size_t n2,
    size_t stride1 = 1,
    size_t stride2 = 1
) {
    int threadId = threadIdx.x + blockDim.x * blockIdx.x;
    if (threadId == 0) {
        printf(
            "GPU: reduceBabyBear %llu %llu %llu\n",
            (uint64_t)n1,
            (uint64_t)n2,
            (uint64_t)(n1 + n2)
        );
    }

    const bn254_t po2 = bn254_t(ALT_BN128_1ls32);
    bn254_t res;
    res.zero();
    // return res;
    if (n2 > 0) {
        for (size_t ii = (n2 - 1) * stride2; true; ii -= stride2) {
            res = res * po2 + bb31_to_bn254(src2[ii]);
            // res += bn254_t::one();
            if (ii < stride2)
                break;  // Prevent underflow
        }
    }
    if (n1 > 0) {
        for (size_t ii = (n1 - 1) * stride1; true; ii -= stride1) {
            res = res * po2 + bb31_to_bn254(src1[ii]);
            // res += bn254_t::one();
            if (ii < stride1)
                break;  // Prevent underflow
        }
    }

    return res;
}

// __device__ void cprow(
//     Matrix<bb31_t>* src,
//     bb31_t* dst,
//     size_t rowIdx,
//     size_t colIdx,
//     size_t n
// ) {
//     if (src->row_major) {
//         for (size_t i = 0; i < n; i++) {
//             dst[i] = src->values[rowIdx * src->width + colIdx + i];
//         }
//     } else {
//         for (size_t i = 0; i < n; i++) {
//             dst[i] = src->values[(colIdx + i) * src->height + rowIdx];
//         }
//     }
// }

template<typename Hasher, typename HasherState>
__device__ void
absorbRow(Hasher hasher, Matrix<bb31_t>* in, int row_idx, HasherState* state) {
    if (threadIdx.x + blockDim.x * blockIdx.x == 0) {
        printf("\nGPU: absorbRow %llu\n", (uint64_t)in->width);
        printf("\nGPU: overhang size %llu\n", (uint64_t)state->overhangSize);
    }

    bb31_t* row_ptr;
    size_t stride;
    if (in->row_major) {
        row_ptr = &in->values[row_idx * in->width];
        stride = 1;
    } else {
        row_ptr = &in->values[row_idx];
        stride = in->height;
    }

    int colIdx = 0;

    if (state->overhangSize > 0) {
        if (state->overhangSize + in->width < ARRAY_SIZE) {
            // Overhang + row is smaller than ARRAY_SIZE, copy row into overhang and return
            for (size_t i = 0; i < in->width; i++) {
                state->overhang[state->overhangSize + i] = row_ptr[i * stride];
            }
            state->overhangSize += in->width;
            return;
        } else {
            // Overhang + row is larger or equal to ARRAY_SIZE, create bn254_t value from overhang and row
            colIdx = ARRAY_SIZE - state->overhangSize;
            if (threadIdx.x + blockDim.x * blockIdx.x == 0)
                printf("\nGPU: reduceBabyBear 1\n");
            bn254_t value = reduceBabyBear(
                state->overhang,
                row_ptr,
                state->overhangSize,
                colIdx,
                1,
                stride
            );
            state->overhangSize = 0;
            hasher.absorb(&value, 1, state);
        }
    }

    // TODO: cleaner! size_t vs int

    while (colIdx + ARRAY_SIZE <= in->width) {
        if (threadIdx.x + blockDim.x * blockIdx.x == 0)
            printf("\nGPU: reduceBabyBear 2\n");
        bn254_t value = reduceBabyBear(
            row_ptr + colIdx * stride,
            nullptr,
            ARRAY_SIZE,
            0,
            stride,
            0
        );
        hasher.absorb(&value, 1, state);
        colIdx += ARRAY_SIZE;
    }

    if (colIdx < in->width) {
        // Copy remaining row into overhang
        for (size_t i = 0; i < in->width - colIdx; i++) {
            state->overhang[state->overhangSize + i] =
                row_ptr[(colIdx + i) * stride];
        }
        state->overhangSize = in->width - colIdx;
    }
}

}  // namespace poseidon2_bn254_3
