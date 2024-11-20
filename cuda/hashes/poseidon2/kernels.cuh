#include "poseidon2.cuh"
#include "poseidon2_bb31_16.cuh"
#include "poseidon2_bn254_3.cuh"

namespace poseidon2_baby_bear_kernels {
using HashParams = poseidon2_bb31_16::BabyBear;

__global__ void permute(
    bb31_t (*in)[HashParams::WIDTH],
    bb31_t (*out)[HashParams::WIDTH],
    size_t n
) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    poseidon2::BabyBearHasher hasher;
    hasher.permute(in[idx], out[idx]);
}

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
    poseidon2::BabyBearHasher hasher;
    hasher.compress(left[idx], right[idx], out[idx]);
}

__global__ void
hash(bb31_t* in, int nIn, bb31_t (*out)[HashParams::DIGEST_WIDTH], int n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    poseidon2::BabyBearHasher hasher;
    hasher.hash(in + idx * nIn, nIn, out[idx]);
}

}  // namespace poseidon2_baby_bear_kernels

namespace poseidon2_bn254_kernels {
using HashParams = poseidon2_bn254_3::Bn254;

__global__ void permute(
    poseidon2::Bn254Hasher hasher,
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

__global__ void compress(
    poseidon2::Bn254Hasher hasher,
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

__global__ void hash(
    poseidon2::Bn254Hasher hasher,
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

extern "C" namespace poseidon2_baby_bear_16_gpu {
    using HashParams = poseidon2_bb31_16::BabyBear;

    extern "C" void permute_baby_bear(
        bb31_t(*in)[HashParams::WIDTH],
        bb31_t(*out)[HashParams::WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        printf("Calling permute_baby_bear\n");
        printf("n is %d\n", n);
        printf("nBlocks is %d\n", nBlocks);
        printf("nThreadsPerBlock is %d\n", nThreadsPerBlock);

        poseidon2_baby_bear_kernels::permute<<<nBlocks, nThreadsPerBlock>>>(
            in,
            out,
            n
        );
    }

    extern "C" void compress_baby_bear(
        bb31_t(*left)[HashParams::DIGEST_WIDTH],
        bb31_t(*right)[HashParams::DIGEST_WIDTH],
        bb31_t(*out)[HashParams::DIGEST_WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2_baby_bear_kernels::compress<<<nBlocks, nThreadsPerBlock>>>(
            left,
            right,
            out,
            n
        );
    }

    extern "C" void hash_baby_bear(
        bb31_t * in,
        size_t nIn,
        bb31_t(*out)[HashParams::DIGEST_WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2_baby_bear_kernels::hash<<<nBlocks, nThreadsPerBlock>>>(
            in,
            nIn,
            out,
            n
        );
    }
}  // namespace poseidon2_baby_bear_16_gpu

extern "C" namespace poseidon2_bn254_3_gpu {
    using namespace poseidon2;

    using HashParams = poseidon2_bn254_3::Bn254;
    using Hasher_t = Bn254Hasher;
    using F_t = typename HashParams::F_t;
    using pF_t = typename HashParams::pF_t;

    extern "C" void permute_bn254(
        F_t(*in)[HashParams::WIDTH],
        F_t(*out)[HashParams::WIDTH],
        pF_t * internalRoundConstants,
        pF_t * externalRoundConstants,
        pF_t * matInternalDiagM1,
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        Hasher_t hasher;
        hasher.setInternalRoundConstants(internalRoundConstants);
        hasher.setExternalRoundConstants(externalRoundConstants);
        hasher.setMatInternalDiagM1(matInternalDiagM1);
        poseidon2_bn254_kernels::permute<<<nBlocks, nThreadsPerBlock>>>(
            hasher,
            in,
            out,
            n
        );
    }

    extern "C" void compress_bn254(
        F_t(*left)[HashParams::DIGEST_WIDTH],
        F_t(*right)[HashParams::DIGEST_WIDTH],
        F_t(*out)[HashParams::DIGEST_WIDTH],
        pF_t * internalRoundConstants,
        pF_t * externalRoundConstants,
        pF_t * matInternalDiagM1,
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        Hasher_t hasher;
        hasher.setInternalRoundConstants(internalRoundConstants);
        hasher.setExternalRoundConstants(externalRoundConstants);
        hasher.setMatInternalDiagM1(matInternalDiagM1);
        poseidon2_bn254_kernels::compress<<<nBlocks, nThreadsPerBlock>>>(
            hasher,
            left,
            right,
            out,
            n
        );
    }

    extern "C" void hash_bn254(
        F_t * in,
        size_t nIn,
        F_t(*out)[HashParams::DIGEST_WIDTH],
        pF_t * internalRoundConstants,
        pF_t * externalRoundConstants,
        pF_t * matInternalDiagM1,
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        Hasher_t hasher;
        hasher.setInternalRoundConstants(internalRoundConstants);
        hasher.setExternalRoundConstants(externalRoundConstants);
        hasher.setMatInternalDiagM1(matInternalDiagM1);
        poseidon2_bn254_kernels::hash<<<nBlocks, nThreadsPerBlock>>>(
            hasher,
            in,
            nIn,
            out,
            n
        );
    }
}  // namespace poseidon2_bn254_3_gpu
