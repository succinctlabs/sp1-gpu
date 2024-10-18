#include <bit>

#include "moongate_cuda_cbindgen.hpp"
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

namespace poseidon2_baby_bear_16_gpu {
using HashParams = poseidon2_bb31_16::BabyBear;

inline void permute_baby_bear(
    bb31_t (*in)[HashParams::WIDTH],
    bb31_t (*out)[HashParams::WIDTH],
    size_t n,
    size_t nBlocks,
    size_t nThreadsPerBlock
) {
    poseidon2_baby_bear_kernels::permute<<<nBlocks, nThreadsPerBlock>>>(
        in,
        out,
        n
    );
}

inline void compress_baby_bear(
    bb31_t (*left)[HashParams::DIGEST_WIDTH],
    bb31_t (*right)[HashParams::DIGEST_WIDTH],
    bb31_t (*out)[HashParams::DIGEST_WIDTH],
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

inline void hash_baby_bear(
    bb31_t* in,
    size_t nIn,
    bb31_t (*out)[HashParams::DIGEST_WIDTH],
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

namespace poseidon2_bn254_3_gpu {
using namespace poseidon2;

using HashParams = poseidon2_bn254_3::Bn254;
using Hasher_t = Bn254Hasher;
using F_t = typename HashParams::F_t;
using pF_t = typename HashParams::pF_t;

inline void permute_bn254(
    F_t (*in)[HashParams::WIDTH],
    F_t (*out)[HashParams::WIDTH],
    pF_t* internalRoundConstants,
    pF_t* externalRoundConstants,
    pF_t* matInternalDiagM1,
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

inline void compress_bn254(
    F_t (*left)[HashParams::DIGEST_WIDTH],
    F_t (*right)[HashParams::DIGEST_WIDTH],
    F_t (*out)[HashParams::DIGEST_WIDTH],
    pF_t* internalRoundConstants,
    pF_t* externalRoundConstants,
    pF_t* matInternalDiagM1,
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

inline void hash_bn254(
    F_t* in,
    size_t nIn,
    F_t (*out)[HashParams::DIGEST_WIDTH],
    pF_t* internalRoundConstants,
    pF_t* externalRoundConstants,
    pF_t* matInternalDiagM1,
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

namespace moongate {

void permute_baby_bear(
    const BabyBear (*input)[BB31_WIDTH],
    BabyBear (*output)[BB31_WIDTH],
    uintptr_t n,
    uintptr_t n_blocks,
    uintptr_t n_threads_per_block
) {
    // TODO propagate const instead of casting it away
    poseidon2_baby_bear_16_gpu::permute_baby_bear(
        std::bit_cast<bb31_t(*)[BB31_WIDTH]>(input),
        std::bit_cast<bb31_t(*)[BB31_WIDTH]>(output),
        n,
        n_blocks,
        n_threads_per_block
    );
}

void compress_baby_bear(
    const BabyBear (*left)[BB31_DIGEST_WIDTH],
    const BabyBear (*right)[BB31_DIGEST_WIDTH],
    BabyBear (*output)[BB31_DIGEST_WIDTH],
    uintptr_t n,
    uintptr_t n_blocks,
    uintptr_t n_threads_per_block
) {
    // TODO propagate const instead of casting it away
    poseidon2_baby_bear_16_gpu::compress_baby_bear(
        std::bit_cast<bb31_t(*)[BB31_DIGEST_WIDTH]>(left),
        std::bit_cast<bb31_t(*)[BB31_DIGEST_WIDTH]>(right),
        std::bit_cast<bb31_t(*)[BB31_DIGEST_WIDTH]>(output),
        n,
        n_blocks,
        n_threads_per_block
    );
}

void hash_baby_bear(
    const BabyBear* input,
    uintptr_t n_input,
    BabyBear (*output)[BB31_DIGEST_WIDTH],
    uintptr_t n,
    uintptr_t n_blocks,
    uintptr_t n_threads_per_block
) {
    // TODO propagate const instead of casting it away
    poseidon2_baby_bear_16_gpu::hash_baby_bear(
        (bb31_t*)input,
        n_input,
        (bb31_t(*)[BB31_DIGEST_WIDTH])output,
        n,
        n_blocks,
        n_threads_per_block
    );
}

void permute_bn254(
    const Bn254Fr (*input)[BN254_WIDTH],
    Bn254Fr (*output)[BN254_WIDTH],
    const Bn254Fr* internal_round_constants,
    const Bn254Fr (*external_round_constants)[BN254_WIDTH],
    const Bn254Fr* diffusion_matrix_m1,
    uintptr_t n,
    uintptr_t n_blocks,
    uintptr_t n_threads_per_block
) {
    poseidon2_bn254_3_gpu::permute_bn254(
        std::bit_cast<bn254_t(*)[BN254_WIDTH]>(input),
        std::bit_cast<bn254_t(*)[BN254_WIDTH]>(output),
        std::bit_cast<bn254_t*>(internal_round_constants),
        std::bit_cast<bn254_t*>(external_round_constants),
        std::bit_cast<bn254_t*>(diffusion_matrix_m1),
        n,
        n_blocks,
        n_threads_per_block
    );
}

void compress_bn254(
    const Bn254Fr (*left)[BN254_DIGEST_WIDTH],
    const Bn254Fr (*right)[BN254_DIGEST_WIDTH],
    Bn254Fr (*output)[BN254_DIGEST_WIDTH],
    const Bn254Fr* internal_round_constants,
    const Bn254Fr (*external_round_constants)[BN254_WIDTH],
    const Bn254Fr* diffusion_matrix_m1,
    uintptr_t n,
    uintptr_t n_blocks,
    uintptr_t n_threads_per_block
) {
    poseidon2_bn254_3_gpu::compress_bn254(
        std::bit_cast<bn254_t(*)[BN254_DIGEST_WIDTH]>(left),
        std::bit_cast<bn254_t(*)[BN254_DIGEST_WIDTH]>(right),
        std::bit_cast<bn254_t(*)[BN254_DIGEST_WIDTH]>(output),
        std::bit_cast<bn254_t*>(internal_round_constants),
        std::bit_cast<bn254_t*>(external_round_constants),
        std::bit_cast<bn254_t*>(diffusion_matrix_m1),
        n,
        n_blocks,
        n_threads_per_block
    );
}

void hash_bn254(
    const Bn254Fr* input,
    uintptr_t n_input,
    Bn254Fr (*output)[BN254_DIGEST_WIDTH],
    const Bn254Fr* internal_round_constants,
    const Bn254Fr (*external_round_constants)[BN254_WIDTH],
    const Bn254Fr* diffusion_matrix_m1,
    uintptr_t n,
    uintptr_t n_blocks,
    uintptr_t n_threads_per_block
) {
    poseidon2_bn254_3_gpu::hash_bn254(
        std::bit_cast<bn254_t*>(input),
        n_input,
        std::bit_cast<bn254_t(*)[BN254_DIGEST_WIDTH]>(output),
        std::bit_cast<bn254_t*>(internal_round_constants),
        std::bit_cast<bn254_t*>(external_round_constants),
        std::bit_cast<bn254_t*>(diffusion_matrix_m1),
        n,
        n_blocks,
        n_threads_per_block
    );
}

}  // namespace moongate
