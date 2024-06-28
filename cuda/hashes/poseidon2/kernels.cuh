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
    using HashParams = poseidon2_bb31_16::BabyBear;
    using F_t = typename HashParams::F_t;

    extern "C" void permute_bb31(
        F_t(*in)[HashParams::WIDTH],
        F_t(*out)[HashParams::WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2_bb31_kernels::permute<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(in, out, n);
    }

    extern "C" void compress_bb31(
        F_t(*left)[HashParams::DIGEST_WIDTH],
        F_t(*right)[HashParams::DIGEST_WIDTH],
        F_t(*out)[HashParams::DIGEST_WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2_bb31_kernels::compress<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(left, right, out, n);
    }

    extern "C" void hash_bb31(
        F_t * in,
        size_t nIn,
        F_t(*out)[HashParams::DIGEST_WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2_bb31_kernels::hash<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(in, nIn, out, n);
    }
}  // namespace poseidon2_bb31_16_gpu

extern "C" namespace poseidon2_bn254_3_gpu {
    using HashParams = poseidon2_bn254_3::Bn254;
    using F_t = typename HashParams::F_t;
    using pF_t = typename HashParams::pF_t;

    extern "C" void permute_bn254(
        F_t(*in)[HashParams::WIDTH],
        F_t(*out)[HashParams::WIDTH],
        pF_t(*internalRoundConstants)[HashParams::ROUNDS_P],
        pF_t(*externalRoundConstants)[HashParams::ROUNDS_F * HashParams::WIDTH],
        pF_t(*matInternalDiagM1)[HashParams::WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2::DynamicHasher<HashParams> hasher;
        hasher.setInternalRoundConstants(internalRoundConstants);
        hasher.setExternalRoundConstants(externalRoundConstants);
        hasher.setMatInternalDiagM1(matInternalDiagM1);
        poseidon2_bn254_kernels::permute<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(hasher, in, out, n);
    }

    extern "C" void compress_bn254(
        F_t(*left)[HashParams::DIGEST_WIDTH],
        F_t(*right)[HashParams::DIGEST_WIDTH],
        F_t(*out)[HashParams::DIGEST_WIDTH],
        pF_t(*internalRoundConstants)[HashParams::ROUNDS_P],
        pF_t(*externalRoundConstants)[HashParams::ROUNDS_F * HashParams::WIDTH],
        pF_t(*matInternalDiagM1)[HashParams::WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2::DynamicHasher<HashParams> hasher;
        hasher.setInternalRoundConstants(internalRoundConstants);
        hasher.setExternalRoundConstants(externalRoundConstants);
        hasher.setMatInternalDiagM1(matInternalDiagM1);
        poseidon2_bn254_kernels::compress<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(hasher, left, right, out, n);
    }

    extern "C" void hash_bn254(
        F_t * in,
        size_t nIn,
        F_t(*out)[HashParams::DIGEST_WIDTH],
        pF_t(*internalRoundConstants)[HashParams::ROUNDS_P],
        pF_t(*externalRoundConstants)[HashParams::ROUNDS_F * HashParams::WIDTH],
        pF_t(*matInternalDiagM1)[HashParams::WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2::DynamicHasher<HashParams> hasher;
        hasher.setInternalRoundConstants(internalRoundConstants);
        hasher.setExternalRoundConstants(externalRoundConstants);
        hasher.setMatInternalDiagM1(matInternalDiagM1);
        poseidon2_bn254_kernels::hash<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(hasher, in, nIn, out, n);
    }
}  // namespace poseidon2_bn254_3_gpu
