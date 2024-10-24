#pragma once

#include <stdio.h>

#include <algorithm>
#include <bit>

#include "../fields/bb31_t.cuh"
#include "../hashes/poseidon2/kernels.cuh"
#include "../matrix/matrix.cuh"
#include "moongate_cuda_cbindgen.hpp"

template<
    typename HashParams,
    typename Hasher_t,
    typename HasherState_t,
    typename Matrix_t>
__device__ void firstDigestLayer(
    Hasher_t hasher,
    Matrix_t* tallestMatrices,
    size_t nTallestMatrices,
    typename HashParams::F_t (*digests)[HashParams::DIGEST_WIDTH]
) {
    int rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rowIdx >= tallestMatrices[0].height) {
        return;
    }

    HasherState_t state;

    for (int i = 0; i < nTallestMatrices; i++) {
        state.absorbRow(hasher, &tallestMatrices[i], rowIdx);
    }
    state.finalize(hasher, digests[rowIdx]);
}

template<
    typename HashParams,
    typename Hasher_t,
    typename HasherState_t,
    typename Matrix_t>
__device__ void compressAndInject(
    Hasher_t hasher,
    typename HashParams::F_t (*prevLayer)[HashParams::DIGEST_WIDTH],
    Matrix_t* matricesToInject,
    size_t nMatricesToInject,
    typename HashParams::F_t (*nextDigests)[HashParams::DIGEST_WIDTH],
    size_t layerLen
) {
    int rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rowIdx >= layerLen) {
        return;
    }

    using F_t = typename HashParams::F_t;

    if (nMatricesToInject == 0) {
        hasher.compress(
            prevLayer[rowIdx * 2],
            prevLayer[rowIdx * 2 + 1],
            nextDigests[rowIdx]
        );
        return;
    }

    size_t nextLen = matricesToInject[0].height;

    F_t defaultDigest[HashParams::DIGEST_WIDTH];
    for (int i = 0; i < HashParams::DIGEST_WIDTH; i++) {
        defaultDigest[i].set_to_zero();
    }

    F_t digest[HashParams::DIGEST_WIDTH];
    hasher.compress(prevLayer[rowIdx * 2], prevLayer[rowIdx * 2 + 1], digest);

    if (rowIdx < nextLen) {
        F_t tallestDigest[HashParams::DIGEST_WIDTH];
        HasherState_t state;
        for (int i = 0; i < nMatricesToInject; i++) {
            state.absorbRow(hasher, &matricesToInject[i], rowIdx);
        }
        state.finalize(hasher, tallestDigest);
        hasher.compress(digest, tallestDigest, nextDigests[rowIdx]);
    } else {
        hasher.compress(
            (F_t*)digest,
            (F_t*)defaultDigest,
            (F_t*)nextDigests[rowIdx]
        );
    }
}

namespace merkle_tree_kernels_baby_bear_16 {
using namespace poseidon2;

using HashParams = poseidon2_bb31_16::BabyBear;
using Hasher_t = BabyBearHasher;
using HasherState_t = BabyBearHasherState;
using Matrix_t = Matrix<bb31_t>;

__launch_bounds__(256, 2) __global__ void firstDigestLayer(
    Matrix_t* tallestMatrices,
    size_t nTallestMatrices,
    bb31_t (*digests)[HashParams::DIGEST_WIDTH]
) {
    Hasher_t hasher;
    ::firstDigestLayer<HashParams, Hasher_t, HasherState_t, Matrix_t>(
        hasher,
        tallestMatrices,
        nTallestMatrices,
        digests
    );
}

__launch_bounds__(128, 1) __global__ void compressAndInject(
    bb31_t (*prevLayer)[HashParams::DIGEST_WIDTH],
    Matrix_t* matricesToInject,
    size_t nMatricesToInject,
    bb31_t (*nextDigests)[HashParams::DIGEST_WIDTH],
    size_t layerLen
) {
    Hasher_t hasher;
    ::compressAndInject<HashParams, Hasher_t, HasherState_t, Matrix_t>(
        hasher,
        prevLayer,
        matricesToInject,
        nMatricesToInject,
        nextDigests,
        layerLen
    );
}

}  // namespace merkle_tree_kernels_baby_bear_16

namespace merkle_tree_kernels_bn254_3 {
using namespace poseidon2;

using HashParams = poseidon2_bn254_3::Bn254;
using Hasher_t = Bn254Hasher;
using HasherState_t = Bn254HasherState;
using Matrix_t = Matrix<bb31_t>;

__launch_bounds__(256, 2) __global__ void firstDigestLayer(
    Hasher_t hasher,
    Matrix_t* tallestMatrices,
    size_t nTallestMatrices,
    bn254_t (*digests)[HashParams::DIGEST_WIDTH]
) {
    ::firstDigestLayer<HashParams, Hasher_t, HasherState_t, Matrix_t>(
        hasher,
        tallestMatrices,
        nTallestMatrices,
        digests
    );
}

__launch_bounds__(128, 1) __global__ void compressAndInject(
    Hasher_t hasher,
    bn254_t (*prevLayer)[HashParams::DIGEST_WIDTH],
    Matrix_t* matricesToInject,
    size_t nMatricesToInject,
    bn254_t (*nextDigests)[HashParams::DIGEST_WIDTH],
    size_t layerLen
) {
    ::compressAndInject<HashParams, Hasher_t, HasherState_t, Matrix_t>(
        hasher,
        prevLayer,
        matricesToInject,
        nMatricesToInject,
        nextDigests,
        layerLen
    );
}

namespace column_major {}

}  // namespace merkle_tree_kernels_bn254_3

namespace merkle_tree_baby_bear_16_gpu {
using HashParams = poseidon2_bb31_16::BabyBear;
using F_t = typename HashParams::F_t;
using pF_t = typename HashParams::pF_t;
using Matrix_t = Matrix<bb31_t>;

inline void first_digest_layer_baby_bear(
    Matrix_t* tallestMatrices,
    size_t nTallestMatrices,
    F_t (*digests)[HashParams::DIGEST_WIDTH],
    size_t max_height
) {
    size_t blockSize = std::min(max_height, static_cast<size_t>(256));
    size_t gridSize = (max_height - 1) / blockSize + 1;
    merkle_tree_kernels_baby_bear_16::firstDigestLayer<<<gridSize, blockSize>>>(
        tallestMatrices,
        nTallestMatrices,
        digests
    );
}

inline void compress_and_inject_baby_bear(
    F_t (*prevLayer)[HashParams::DIGEST_WIDTH],
    Matrix_t* matricesToInject,
    size_t nMatricesToInject,
    F_t (*nextDigests)[HashParams::DIGEST_WIDTH],
    size_t layerLen
) {
    size_t blockSize = std::min(layerLen, static_cast<size_t>(128));
    size_t gridSize = (layerLen - 1) / blockSize + 1;
    merkle_tree_kernels_baby_bear_16::
        compressAndInject<<<gridSize, blockSize>>>(
            prevLayer,
            matricesToInject,
            nMatricesToInject,
            nextDigests,
            layerLen
        );
}
}  // namespace merkle_tree_baby_bear_16_gpu

namespace merkle_tree_bn254_3_gpu {
using HashParams = poseidon2_bn254_3::Bn254;
using F_t = typename HashParams::F_t;
using pF_t = typename HashParams::pF_t;
using Matrix_t = Matrix<bb31_t>;

inline void first_digest_layer_bn254(
    Matrix_t* tallestMatrices,
    size_t nTallestMatrices,
    F_t (*digests)[HashParams::DIGEST_WIDTH],
    pF_t* internalRoundConstants,
    pF_t* externalRoundConstants,
    pF_t* matInternalDiagM1,
    size_t max_height
) {
    size_t blockSize = std::min(max_height, static_cast<size_t>(256));
    size_t gridSize = (max_height - 1) / blockSize + 1;
    poseidon2::Bn254Hasher hasher;
    hasher.setInternalRoundConstants(internalRoundConstants);
    hasher.setExternalRoundConstants(externalRoundConstants);
    hasher.setMatInternalDiagM1(matInternalDiagM1);
    merkle_tree_kernels_bn254_3::firstDigestLayer<<<gridSize, blockSize>>>(
        hasher,
        tallestMatrices,
        nTallestMatrices,
        digests
    );
}

inline void compress_and_inject_bn254(
    F_t (*prevLayer)[HashParams::DIGEST_WIDTH],
    Matrix_t* matricesToInject,
    size_t nMatricesToInject,
    F_t (*nextDigests)[HashParams::DIGEST_WIDTH],
    pF_t* internalRoundConstants,
    pF_t* externalRoundConstants,
    pF_t* matInternalDiagM1,
    size_t layerLen
) {
    size_t blockSize = std::min(layerLen, static_cast<size_t>(128));
    size_t gridSize = (layerLen - 1) / blockSize + 1;
    poseidon2::Bn254Hasher hasher;
    hasher.setInternalRoundConstants(internalRoundConstants);
    hasher.setExternalRoundConstants(externalRoundConstants);
    hasher.setMatInternalDiagM1(matInternalDiagM1);
    merkle_tree_kernels_bn254_3::compressAndInject<<<gridSize, blockSize>>>(
        hasher,
        prevLayer,
        matricesToInject,
        nMatricesToInject,
        nextDigests,
        layerLen
    );
}
}  // namespace merkle_tree_bn254_3_gpu

namespace moongate {

void first_digest_layer_baby_bear(
    const MatrixViewDevice<BabyBear>* tallest_matrices,
    uintptr_t n_tallest_matrices,
    BabyBear (*digests)[BB31_DIGEST_WIDTH],
    uintptr_t max_height
) {
    merkle_tree_baby_bear_16_gpu::first_digest_layer_baby_bear(
        std::bit_cast<Matrix<bb31_t>*>(tallest_matrices),
        n_tallest_matrices,
        std::bit_cast<bb31_t(*)[BB31_DIGEST_WIDTH]>(digests),
        max_height
    );
}

void compress_and_inject_baby_bear(
    const BabyBear (*prev_layer)[BB31_DIGEST_WIDTH],
    const MatrixViewDevice<BabyBear>* matrices_to_inject,
    uintptr_t n_matrices_to_inject,
    BabyBear (*next_digests)[BB31_DIGEST_WIDTH],
    uintptr_t layer_len
) {
    merkle_tree_baby_bear_16_gpu::compress_and_inject_baby_bear(

        std::bit_cast<bb31_t(*)[BB31_DIGEST_WIDTH]>(prev_layer),
        std::bit_cast<Matrix<bb31_t>*>(matrices_to_inject),
        n_matrices_to_inject,
        std::bit_cast<bb31_t(*)[BB31_DIGEST_WIDTH]>(next_digests),
        layer_len
    );
}

void first_digest_layer_bn254(
    const MatrixViewDevice<BabyBear>* tallest_matrices,
    uintptr_t n_tallest_matrices,
    Bn254Fr (*digests)[BN254_DIGEST_WIDTH],
    const Bn254Fr* internal_round_constants,
    const Bn254Fr (*external_round_constants)[BN254_WIDTH],
    const Bn254Fr* diffusion_matrix_m1,
    uintptr_t max_height
) {
    merkle_tree_bn254_3_gpu::first_digest_layer_bn254(
        std::bit_cast<Matrix<bb31_t>*>(tallest_matrices),
        n_tallest_matrices,
        std::bit_cast<bn254_t(*)[BN254_DIGEST_WIDTH]>(digests),
        std::bit_cast<bn254_t*>(internal_round_constants),
        std::bit_cast<bn254_t*>(external_round_constants),
        std::bit_cast<bn254_t*>(diffusion_matrix_m1),
        max_height
    );
}

void compress_and_inject_bn254(
    const Bn254Fr (*prev_layer)[BN254_DIGEST_WIDTH],
    const MatrixViewDevice<BabyBear>* matrices_to_inject,
    uintptr_t n_matrices_to_inject,
    Bn254Fr (*next_digests)[BN254_DIGEST_WIDTH],
    const Bn254Fr* internal_round_constants,
    const Bn254Fr (*external_round_constants)[BN254_WIDTH],
    const Bn254Fr* diffusion_matrix_m1,
    uintptr_t max_height
) {
    merkle_tree_bn254_3_gpu::compress_and_inject_bn254(
        std::bit_cast<bn254_t(*)[BN254_DIGEST_WIDTH]>(prev_layer),
        (Matrix<bb31_t>*)matrices_to_inject,
        n_matrices_to_inject,
        std::bit_cast<bn254_t(*)[BN254_DIGEST_WIDTH]>(next_digests),
        std::bit_cast<bn254_t*>(internal_round_constants),
        std::bit_cast<bn254_t*>(external_round_constants),
        std::bit_cast<bn254_t*>(diffusion_matrix_m1),
        max_height
    );
}

}  // namespace moongate
