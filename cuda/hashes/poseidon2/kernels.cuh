#include "poseidon2.cuh"
#include "poseidon2_bb31_16.cuh"

namespace poseidon2_bb31_16_kernels {
using namespace poseidon2_bb31_16;

__global__ void permute(bb31_t (*in)[WIDTH], bb31_t (*out)[WIDTH], size_t n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) return;

    Hasher hasher = newBabyBear16();
    hasher.permute(in[idx], out[idx]);
}

__global__ void compress(bb31_t (*left)[poseidon2::DIGEST_WIDTH],
                         bb31_t (*right)[poseidon2::DIGEST_WIDTH],
                         bb31_t (*out)[poseidon2::DIGEST_WIDTH], size_t n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) return;

    Hasher hasher = newBabyBear16();
    hasher.compress(left[idx], right[idx], out[idx]);
}

__global__ void hash(bb31_t* in, int nIn,
                     bb31_t (*out)[poseidon2::DIGEST_WIDTH], int n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) return;

    Hasher hasher = newBabyBear16();
    hasher.hash(in + idx * nIn, nIn, out[idx]);
}
}  // namespace poseidon2_bb31_16_kernels

extern "C" namespace poseidon2_bb31_16_gpu {
using namespace poseidon2_bb31_16;

extern "C" void permute(bb31_t (*in)[WIDTH], bb31_t (*out)[WIDTH], size_t n,
                        size_t nBlocks, size_t nThreadsPerBlock) {
    poseidon2_bb31_16_kernels::permute<<<nBlocks, nThreadsPerBlock>>>(in, out, n);
}

extern "C" void compress(bb31_t (*left)[poseidon2::DIGEST_WIDTH],
                         bb31_t (*right)[poseidon2::DIGEST_WIDTH],
                         bb31_t (*out)[poseidon2::DIGEST_WIDTH], size_t n,
                         size_t nBlocks, size_t nThreadsPerBlock) {
    poseidon2_bb31_16_kernels::compress<<<nBlocks, nThreadsPerBlock>>>(left, right, out, n);
}

extern "C" void hash(bb31_t* in, size_t nIn,
                     bb31_t (*out)[poseidon2::DIGEST_WIDTH], size_t n,
                     size_t nBlocks, size_t nThreadsPerBlock) {
    poseidon2_bb31_16_kernels::hash<<<nBlocks, nThreadsPerBlock>>>(in, nIn, out, n);
}
}  // namespace poseidon2_bb31_16_gpu
