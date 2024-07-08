#pragma once

#include <stdio.h>

#include "../fields/bb31_t.cuh"
#include "../hashes/poseidon2/kernels.cuh"
#include "../matrix/matrix.cuh"

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
    size_t nPrevLayer,
    Matrix_t* matricesToInject,
    size_t nMatricesToInject,
    typename HashParams::F_t (*nextDigests)[HashParams::DIGEST_WIDTH]
) {
    int rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rowIdx >= nPrevLayer / 2) {
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
    // size_t nextLenPadded = nPrevLayer / 2;

    F_t defaultDigest[HashParams::DIGEST_WIDTH];
    for (int i = 0; i < HashParams::DIGEST_WIDTH; i++) {
        defaultDigest[i].zero();
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

__global__ void firstDigestLayer(
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

__global__ void compressAndInject(
    bb31_t (*prevLayer)[HashParams::DIGEST_WIDTH],
    size_t nPrevLayer,
    Matrix_t* matricesToInject,
    size_t nMatricesToInject,
    bb31_t (*nextDigests)[HashParams::DIGEST_WIDTH]
) {
    Hasher_t hasher;
    ::compressAndInject<HashParams, Hasher_t, HasherState_t, Matrix_t>(
        hasher,
        prevLayer,
        nPrevLayer,
        matricesToInject,
        nMatricesToInject,
        nextDigests
    );
}

}  // namespace merkle_tree_kernels_baby_bear_16

namespace merkle_tree_kernels_bn254_3 {
using namespace poseidon2;

using HashParams = poseidon2_bn254_3::Bn254;
using Hasher_t = Bn254Hasher;
using HasherState_t = Bn254HasherState;
using Matrix_t = Matrix<bb31_t>;

__global__ void firstDigestLayer(
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

__global__ void compressAndInject(
    Hasher_t hasher,
    bn254_t (*prevLayer)[HashParams::DIGEST_WIDTH],
    size_t nPrevLayer,
    Matrix_t* matricesToInject,
    size_t nMatricesToInject,
    bn254_t (*nextDigests)[HashParams::DIGEST_WIDTH]
) {
    ::compressAndInject<HashParams, Hasher_t, HasherState_t, Matrix_t>(
        hasher,
        prevLayer,
        nPrevLayer,
        matricesToInject,
        nMatricesToInject,
        nextDigests
    );
}

namespace column_major {}

}  // namespace merkle_tree_kernels_bn254_3

extern "C" namespace merkle_tree_baby_bear_16_gpu {
    using HashParams = poseidon2_bb31_16::BabyBear;
    using F_t = typename HashParams::F_t;
    using pF_t = typename HashParams::pF_t;
    using Matrix_t = Matrix<bb31_t>;

    extern "C" void first_digest_layer_baby_bear(
        Matrix_t * tallestMatrices,
        size_t nTallestMatrices,
        F_t(*digests)[HashParams::DIGEST_WIDTH],
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        merkle_tree_kernels_baby_bear_16::
            firstDigestLayer<<<nBlocks, nThreadsPerBlock>>>(
                tallestMatrices,
                nTallestMatrices,
                digests
            );
    }

    extern "C" void compress_and_inject_baby_bear(
        F_t(*prevLayer)[HashParams::DIGEST_WIDTH],
        size_t nPrevLayer,
        Matrix_t * matricesToInject,
        size_t nMatricesToInject,
        F_t(*nextDigests)[HashParams::DIGEST_WIDTH],
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        merkle_tree_kernels_baby_bear_16::
            compressAndInject<<<nBlocks, nThreadsPerBlock>>>(
                prevLayer,
                nPrevLayer,
                matricesToInject,
                nMatricesToInject,
                nextDigests
            );
    }
}

extern "C" namespace merkle_tree_bn254_3_gpu {
    using HashParams = poseidon2_bn254_3::Bn254;
    using F_t = typename HashParams::F_t;
    using pF_t = typename HashParams::pF_t;
    using Matrix_t = Matrix<bb31_t>;

    extern "C" void first_digest_layer_bn254(
        Matrix_t * tallestMatrices,
        size_t nTallestMatrices,
        F_t(*digests)[HashParams::DIGEST_WIDTH],
        pF_t * internalRoundConstants,
        pF_t * externalRoundConstants,
        pF_t * matInternalDiagM1,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2::Bn254Hasher hasher;
        hasher.setInternalRoundConstants(internalRoundConstants);
        hasher.setExternalRoundConstants(externalRoundConstants);
        hasher.setMatInternalDiagM1(matInternalDiagM1);
        merkle_tree_kernels_bn254_3::
            firstDigestLayer<<<nBlocks, nThreadsPerBlock>>>(
                hasher,
                tallestMatrices,
                nTallestMatrices,
                digests
            );
    }

    extern "C" void compress_and_inject_bn254(
        F_t(*prevLayer)[HashParams::DIGEST_WIDTH],
        size_t nPrevLayer,
        Matrix_t * matricesToInject,
        size_t nMatricesToInject,
        F_t(*nextDigests)[HashParams::DIGEST_WIDTH],
        pF_t * internalRoundConstants,
        pF_t * externalRoundConstants,
        pF_t * matInternalDiagM1,
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        poseidon2::Bn254Hasher hasher;
        hasher.setInternalRoundConstants(internalRoundConstants);
        hasher.setExternalRoundConstants(externalRoundConstants);
        hasher.setMatInternalDiagM1(matInternalDiagM1);
        merkle_tree_kernels_bn254_3::
            compressAndInject<<<nBlocks, nThreadsPerBlock>>>(
                hasher,
                prevLayer,
                nPrevLayer,
                matricesToInject,
                nMatricesToInject,
                nextDigests
            );
    }
}