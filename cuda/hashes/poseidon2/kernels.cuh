#include "poseidon2.cuh"
#include "poseidon2_bb31_16.cuh"

namespace poseidon2_bb31_16_kernels {
using namespace poseidon2_bb31_16;
__global__ void permute(Hasher hasher, bb31_t (*in)[WIDTH],
                        bb31_t (*out)[WIDTH], size_t n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }

    hasher.permute(in[idx], out[idx]);
}

__global__ void compress(Hasher hasher, bb31_t (*left)[poseidon2::DIGEST_WIDTH],
                         bb31_t (*right)[poseidon2::DIGEST_WIDTH],
                         bb31_t (*out)[poseidon2::DIGEST_WIDTH], size_t n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }

    hasher.compress(left[idx], right[idx], out[idx]);
}

__global__ void hash(Hasher hasher, bb31_t* in, int nIn,
                     bb31_t (*out)[poseidon2::DIGEST_WIDTH], int n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    hasher.hash(in + idx * nIn, nIn, out[idx]);
}
}  // namespace poseidon2_bb31_16_kernels

extern "C" namespace poseidon2_bb31_16_gpu {
    using namespace poseidon2_bb31_16;

    extern "C" void permute(bb31_t(*in)[WIDTH], bb31_t(*out)[WIDTH], size_t n,
                            size_t nBlocks, size_t nThreadsPerBlock) {
        Hasher hasher = newBabyBear16();
        poseidon2_bb31_16_kernels::permute<<<nBlocks, nThreadsPerBlock>>>(
            hasher, in, out, n);
    }

    extern "C" void compress(bb31_t(*left)[poseidon2::DIGEST_WIDTH],
                             bb31_t(*right)[poseidon2::DIGEST_WIDTH],
                             bb31_t(*out)[poseidon2::DIGEST_WIDTH], size_t n,
                             size_t nBlocks, size_t nThreadsPerBlock) {
        Hasher hasher = newBabyBear16();
        poseidon2_bb31_16_kernels::compress<<<nBlocks, nThreadsPerBlock>>>(
            hasher, left, right, out, n);
    }

    extern "C" void hash(bb31_t * in, size_t nIn,
                         bb31_t(*out)[poseidon2::DIGEST_WIDTH], size_t n,
                         size_t nBlocks, size_t nThreadsPerBlock) {
        Hasher hasher = newBabyBear16();
        poseidon2_bb31_16_kernels::hash<<<nBlocks, nThreadsPerBlock>>>(
            hasher, in, nIn, out, n);
    }

}  // namespace poseidon2_bb31_16_gpu
