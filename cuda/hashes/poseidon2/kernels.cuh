#include "poseidon2.cuh"
#include "poseidon2_bb31_16.cuh"
#include "poseidon2_bn254_16.cuh"

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
    Hasher<HashParams> hasher;
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
    Hasher<HashParams> hasher;
    hasher.compress(left[idx], right[idx], out[idx]);
}

template<typename HashParams>
__global__ void
hash(bb31_t* in, int nIn, bb31_t (*out)[HashParams::DIGEST_WIDTH], int n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    Hasher<HashParams> hasher;
    hasher.hash(in + idx * nIn, nIn, out[idx]);
}
}  // namespace poseidon2_bb31_kernels

namespace poseidon2_bn254_kernels {
using namespace poseidon2;

template<typename HashParams>
__global__ void permute(
    bn254_t (*in)[HashParams::WIDTH],
    bn254_t (*out)[HashParams::WIDTH],
    size_t n
) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    Hasher<HashParams> hasher;
    hasher.permute(in[idx], out[idx]);
}

template<typename HashParams>
__global__ void compress(
    bn254_t (*left)[HashParams::DIGEST_WIDTH],
    bn254_t (*right)[HashParams::DIGEST_WIDTH],
    bn254_t (*out)[HashParams::DIGEST_WIDTH],
    size_t n
) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    Hasher<HashParams> hasher;
    hasher.compress(left[idx], right[idx], out[idx]);
}

template<typename HashParams>
__global__ void
hash(bn254_t* in, int nIn, bn254_t (*out)[HashParams::DIGEST_WIDTH], int n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    Hasher<HashParams> hasher;
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

extern "C" namespace poseidon2_bn254_16_gpu {
    using HashParams = poseidon2_bn254_16::BarretoNaehrig16;
    using F = typename HashParams::F;

    extern "C" void permute_bn254(
        F(*in)[HashParams::WIDTH],
        F(*out)[HashParams::WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2_bn254_kernels::permute<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(in, out, n);
    }

    extern "C" void compress_bn254(
        F(*left)[HashParams::DIGEST_WIDTH],
        F(*right)[HashParams::DIGEST_WIDTH],
        F(*out)[HashParams::DIGEST_WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2_bn254_kernels::compress<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(left, right, out, n);
    }

    extern "C" void hash_bn254(
        F * in,
        size_t nIn,
        F(*out)[HashParams::DIGEST_WIDTH],
        size_t n,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2_bn254_kernels::hash<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(in, nIn, out, n);
    }
}  // namespace poseidon2_bn254_16_gpu