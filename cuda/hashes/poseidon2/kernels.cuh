#include "poseidon2.cuh"
#include "poseidon2_bb31_16.cuh"
#include "poseidon2_bn254_3.cuh"

namespace poseidon2_bb31_kernels {
using namespace poseidon2;

template<typename HashParams>
__global__ void permute(
    bb31_t (*in)[HashParams::WIDTH],
    bb31_t (*out)[HashParams::WIDTH],
    size_t n
) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    StaticHasher<HashParams> hasher;
    hasher.permute(in[idx], out[idx]);
}

template<typename HashParams>
__global__ void compress(
    bb31_t (*left)[HashParams::DIGEST_WIDTH],
    bb31_t (*right)[HashParams::DIGEST_WIDTH],
    bb31_t (*out)[HashParams::DIGEST_WIDTH],
    size_t n
) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    StaticHasher<HashParams> hasher;
    hasher.compress(left[idx], right[idx], out[idx]);
}

template<typename HashParams>
__global__ void
hash(bb31_t* in, int nIn, bb31_t (*out)[HashParams::DIGEST_WIDTH], int n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    StaticHasher<HashParams> hasher;
    hasher.hash(in + idx * nIn, nIn, out[idx]);
}
}  // namespace poseidon2_bb31_kernels

namespace poseidon2_bn254_kernels {
using namespace poseidon2;

template<typename HashParams>
__global__ void permute(
    DynamicHasher<HashParams> hasher,
    bn254_t (*in)[HashParams::WIDTH],
    bn254_t (*out)[HashParams::WIDTH],
    size_t n
) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    hasher.permute(in[idx], out[idx]);
}

template<typename HashParams>
__global__ void compress(
    DynamicHasher<HashParams> hasher,
    bn254_t (*left)[HashParams::DIGEST_WIDTH],
    bn254_t (*right)[HashParams::DIGEST_WIDTH],
    bn254_t (*out)[HashParams::DIGEST_WIDTH],
    size_t n
) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    hasher.compress(left[idx], right[idx], out[idx]);
}

template<typename HashParams>
__global__ void hash(
    DynamicHasher<HashParams> hasher,
    bn254_t* in,
    int nIn,
    bn254_t (*out)[HashParams::DIGEST_WIDTH],
    int n
) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    hasher.hash(in + idx * nIn, nIn, out[idx]);
}
}  // namespace poseidon2_bn254_kernels

extern "C" namespace poseidon2_bb31_16_gpu {
    using HashParams = poseidon2_bb31_16::BabyBear16;
    using F = typename HashParams::F;

    extern "C" void permute_bb31(
        F(*in)[HashParams::WIDTH],
        F(*out)[HashParams::WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2_bb31_kernels::permute<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(in, out, n);
    }

    extern "C" void compress_bb31(
        F(*left)[HashParams::DIGEST_WIDTH],
        F(*right)[HashParams::DIGEST_WIDTH],
        F(*out)[HashParams::DIGEST_WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2_bb31_kernels::compress<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(left, right, out, n);
    }

    extern "C" void hash_bb31(
        F * in,
        size_t nIn,
        F(*out)[HashParams::DIGEST_WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2_bb31_kernels::hash<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(in, nIn, out, n);
    }
}  // namespace poseidon2_bb31_16_gpu

extern "C" namespace poseidon2_bn254_3_gpu {
    using HashParams = poseidon2_bn254_3::BarretoNaehrig3;
    using F = typename HashParams::F;

    extern "C" void permute_bn254(
        F(*in)[HashParams::WIDTH],
        F(*out)[HashParams::WIDTH],
        F(*internalRoundConstants)[HashParams::ROUNDS_P],
        F(*externalRoundConstants)[HashParams::ROUNDS_F * HashParams::WIDTH],
        F(*diffusionMatrixM1)[HashParams::WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2::DynamicHasher<HashParams> hasher;
        hasher.setParams(
            internalRoundConstants,
            externalRoundConstants,
            diffusionMatrixM1
        );
        poseidon2_bn254_kernels::permute<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(hasher, in, out, n);
    }

    extern "C" void compress_bn254(
        F(*left)[HashParams::DIGEST_WIDTH],
        F(*right)[HashParams::DIGEST_WIDTH],
        F(*out)[HashParams::DIGEST_WIDTH],
        F(*internalRoundConstants)[HashParams::ROUNDS_P],
        F(*externalRoundConstants)[HashParams::ROUNDS_F * HashParams::WIDTH],
        F(*diffusionMatrixM1)[HashParams::WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2::DynamicHasher<HashParams> hasher;
        hasher.setParams(
            internalRoundConstants,
            externalRoundConstants,
            diffusionMatrixM1
        );
        poseidon2_bn254_kernels::compress<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(hasher, left, right, out, n);
    }

    extern "C" void hash_bn254(
        F * in,
        size_t nIn,
        F(*out)[HashParams::DIGEST_WIDTH],
        F(*internalRoundConstants)[HashParams::ROUNDS_P],
        F(*externalRoundConstants)[HashParams::ROUNDS_F * HashParams::WIDTH],
        F(*diffusionMatrixM1)[HashParams::WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2::DynamicHasher<HashParams> hasher;
        hasher.setParams(
            internalRoundConstants,
            externalRoundConstants,
            diffusionMatrixM1
        );
        poseidon2_bn254_kernels::hash<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(hasher, in, nIn, out, n);
    }
}  // namespace poseidon2_bn254_3_gpu

// namespace sum_bb31_kernels {
// __global__ void sum_bb31(
//     bb31_t (*left)[poseidon2_bb31_16::BabyBear16::WIDTH],
//     bb31_t (*right)[poseidon2_bb31_16::BabyBear16::WIDTH],
//     bb31_t (*out)[poseidon2_bb31_16::BabyBear16::WIDTH],
//     size_t n
// ) {
//     size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
//     if (idx >= n) {
//         return;
//     }
//     for (size_t ii = 0; ii < poseidon2_bb31_16::BabyBear16::WIDTH; ii++) {
//         out[idx][ii] = left[idx][ii] + right[idx][ii];
//     }
// }
// }  // namespace sum_bb31_kernels

// extern "C" namespace sum_bb31_gpu {
//     extern "C" void sum_bb31(
//         bb31_t(*left)[poseidon2_bb31_16::BabyBear16::WIDTH],
//         bb31_t(*right)[poseidon2_bb31_16::BabyBear16::WIDTH],
//         bb31_t(*out)[poseidon2_bb31_16::BabyBear16::WIDTH],
//         size_t n,
//         size_t nBlocks,
//         size_t nThreadsPerBlock
//     ) {
//         sum_bb31_kernels::sum_bb31<<<nBlocks, nThreadsPerBlock>>>(
//             left,
//             right,
//             out,
//             n
//         );
//     }
// }  // namespace sum_bb31_16_gpu

// #include "../../fields/bn254_t.cuh"

// namespace sum_bn254_kernels {
// __global__ void sum_bn254(
//     bn254_t (*left)[16],
//     bn254_t (*right)[16],
//     bn254_t (*out)[16],
//     size_t n
// ) {
//     // const uint32_t ALT_BN128_rone[8] = {
//     //     0x4ffffffb,
//     //     0xac96341c,
//     //     0x9f60cd29,
//     //     0x36fc7695,
//     //     0x7879462e,
//     //     0x666ea36f,
//     //     0x9a07df2f,
//     //     0x0e0a77c1
//     // };
//     const uint32_t ALT_BN128_rone[8] = {/* (1<<256)%P */
//                                         TO_CUDA_T(0xac96341c4ffffffb),
//                                         TO_CUDA_T(0x36fc76959f60cd29),
//                                         TO_CUDA_T(0x666ea36f7879462e),
//                                         TO_CUDA_T(0x0e0a77c19a07df2f)
//     };
//     const bn254_t val = bn254_t(ALT_BN128_rone);
//     // const bn254_t ONE = bn254_t(device::ALT_BN128_rone);
//     // const bn254_t TWO = ONE + ONE;
//     size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
//     if (idx >= n) {
//         return;
//     }
//     for (size_t ii = 0; ii < 16; ii++) {
//         out[idx][ii] = left[idx][ii] + right[idx][ii];
//         out[idx][ii] *= val;
//     }
// }
// }  // namespace sum_bn254_kernels

// extern "C" namespace sum_bn254_gpu {
//     extern "C" void sum_bn254(
//         bn254_t(*left)[16],
//         bn254_t(*right)[16],
//         bn254_t(*out)[16],
//         size_t n,
//         size_t nBlocks,
//         size_t nThreadsPerBlock
//     ) {
//         sum_bn254_kernels::sum_bn254<<<nBlocks, nThreadsPerBlock>>>(
//             left,
//             right,
//             out,
//             n
//         );
//     }
// }  // namespace sum_bn254_gpu