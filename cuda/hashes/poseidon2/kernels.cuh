#include "poseidon2.cuh"
#include "poseidon2_bb31_16.cuh"

namespace poseidon2_kernels {
using namespace poseidon2;

template <typename Params>
__global__ void permute(bb31_t (*in)[Params::WIDTH],
                        bb31_t (*out)[Params::WIDTH], size_t n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) return;
    Hasher<Params> hasher;
    hasher.permute(in[idx], out[idx]);
}

template <typename Params>
__global__ void compress(bb31_t (*left)[Params::DIGEST_WIDTH],
                         bb31_t (*right)[Params::DIGEST_WIDTH],
                         bb31_t (*out)[Params::DIGEST_WIDTH], size_t n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) return;
    Hasher<Params> hasher;
    hasher.compress(left[idx], right[idx], out[idx]);
}

template <typename Params>
__global__ void hash(bb31_t* in, int nIn, bb31_t (*out)[Params::DIGEST_WIDTH],
                     int n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) return;
    Hasher<Params> hasher;
    hasher.hash(in + idx * nIn, nIn, out[idx]);
}
}  // namespace poseidon2_kernels

extern "C" namespace poseidon2_bb31_16_gpu {
    using Params = poseidon2_bb31_16::BabyBear16;
    using F = typename Params::F;

    extern "C" void permute(F(*in)[Params::WIDTH], F(*out)[Params::WIDTH],
                            size_t n, size_t nBlocks, size_t nThreadsPerBlock) {
        poseidon2_kernels::permute<Params>
            <<<nBlocks, nThreadsPerBlock>>>(in, out, n);
    }

    extern "C" void compress(F(*left)[Params::DIGEST_WIDTH],
                             F(*right)[Params::DIGEST_WIDTH],
                             F(*out)[Params::DIGEST_WIDTH], size_t n,
                             size_t nBlocks, size_t nThreadsPerBlock) {
        poseidon2_kernels::compress<Params>
            <<<nBlocks, nThreadsPerBlock>>>(left, right, out, n);
    }

    extern "C" void hash(F * in, size_t nIn, F(*out)[Params::DIGEST_WIDTH],
                         size_t n, size_t nBlocks, size_t nThreadsPerBlock) {
        poseidon2_kernels::hash<Params>
            <<<nBlocks, nThreadsPerBlock>>>(in, nIn, out, n);
    }
}  // namespace poseidon2_bb31_16_gpu
