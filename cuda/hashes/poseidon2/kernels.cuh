#include "poseidon2.cuh"
#include "poseidon2_bb31_16.cuh"

namespace poseidon2_kernels {
using namespace poseidon2;

template <typename HashParams>
__global__ void permute(bb31_t (*in)[HashParams::WIDTH],
                        bb31_t (*out)[HashParams::WIDTH], size_t n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    Hasher<HashParams> hasher;
    hasher.permute(in[idx], out[idx]);
}

template <typename HashParams>
__global__ void compress(bb31_t (*left)[HashParams::DIGEST_WIDTH],
                         bb31_t (*right)[HashParams::DIGEST_WIDTH],
                         bb31_t (*out)[HashParams::DIGEST_WIDTH], size_t n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    Hasher<HashParams> hasher;
    hasher.compress(left[idx], right[idx], out[idx]);
}

template <typename HashParams>
__global__ void hash(bb31_t* in, int nIn,
                     bb31_t (*out)[HashParams::DIGEST_WIDTH], int n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    Hasher<HashParams> hasher;
    hasher.hash(in + idx * nIn, nIn, out[idx]);
}
}  // namespace poseidon2_kernels

extern "C" namespace poseidon2_bb31_16_gpu {
    using HashParams = poseidon2_bb31_16::BabyBear16;
    using F = typename HashParams::F;

    extern "C" void permute(F(*in)[HashParams::WIDTH],
                            F(*out)[HashParams::WIDTH], size_t n,
                            size_t nBlocks, size_t nThreadsPerBlock) {
        poseidon2_kernels::permute<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(in, out, n);
    }

    extern "C" void compress(F(*left)[HashParams::DIGEST_WIDTH],
                             F(*right)[HashParams::DIGEST_WIDTH],
                             F(*out)[HashParams::DIGEST_WIDTH], size_t n,
                             size_t nBlocks, size_t nThreadsPerBlock) {
        poseidon2_kernels::compress<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(left, right, out, n);
    }

    extern "C" void hash(F * in, size_t nIn, F(*out)[HashParams::DIGEST_WIDTH],
                         size_t n, size_t nBlocks, size_t nThreadsPerBlock) {
        poseidon2_kernels::hash<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(in, nIn, out, n);
    }
}  // namespace poseidon2_bb31_16_gpu
