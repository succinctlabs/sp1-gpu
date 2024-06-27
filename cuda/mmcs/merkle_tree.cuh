#pragma once

#include "../fields/bb31_t.cuh"
#include "../hashes/poseidon2/kernels.cuh"
#include "../matrix/matrix.cuh"

#include <stdio.h>

namespace merkle_tree_kernels {
using namespace poseidon2;

template <typename HashParams>
__global__ void firstDigestLayer(
    Matrix<typename HashParams::F_t> *tallestMatrices, size_t nTallestMatrices,
    typename HashParams::F_t (*digests)[HashParams::DIGEST_WIDTH]) {
    int rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rowIdx >= tallestMatrices[0].height) {
        return;
    }

    HasherState<HashParams> state = HasherState<HashParams>();
    StaticHasher<HashParams> hasher;

    for (int i = 0; i < nTallestMatrices; i++) {
        hasher.absorbRow(&tallestMatrices[i], rowIdx, &state);
    }
    hasher.finalize(&state, digests[rowIdx]);
}

template <typename HashParams>
__global__ void compressAndInject(
    typename HashParams::F_t (*prevLayer)[HashParams::DIGEST_WIDTH],
    size_t nPrevLayer, Matrix<typename HashParams::F_t> *matricesToInject,
    size_t nMatricesToInject,
    typename HashParams::F_t (*nextDigests)[HashParams::DIGEST_WIDTH]) {
    int rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rowIdx >= nPrevLayer / 2) {
        return;
    }

    using F_t = typename HashParams::F_t;

    StaticHasher<HashParams> hasher;

    if (nMatricesToInject == 0) {
        hasher.compress(prevLayer[rowIdx * 2], prevLayer[rowIdx * 2 + 1],
                        nextDigests[rowIdx]);
        return;
    }

    size_t nextLen = matricesToInject[0].height;
    // size_t nextLenPadded = nPrevLayer / 2;

    F_t defaultDigest[HashParams::DIGEST_WIDTH] = {F_t(0), F_t(0), F_t(0),
                                                 F_t(0), F_t(0), F_t(0)};

    F_t digest[HashParams::DIGEST_WIDTH];
    hasher.compress(prevLayer[rowIdx * 2], prevLayer[rowIdx * 2 + 1], digest);

    if (rowIdx < nextLen) {
        F_t tallestDigest[HashParams::DIGEST_WIDTH];
        HasherState<HashParams> state = HasherState<HashParams>();
        for (int i = 0; i < nMatricesToInject; i++) {
            hasher.absorbRow(&matricesToInject[i], rowIdx, &state);
        }
        hasher.finalize(&state, tallestDigest);
        hasher.compress(digest, tallestDigest, nextDigests[rowIdx]);
    } else {
        hasher.compress((F_t *)digest, (F_t *)defaultDigest,
                        (F_t *)nextDigests[rowIdx]);
    }
}

namespace column_major {}
}  // namespace merkle_tree_kernels

extern "C" namespace merkle_tree_gpu {
    using HashParams = poseidon2_bb31_16::BabyBear;
    using F_t = typename HashParams::F_t;

    extern "C" void first_digest_layer(
        Matrix<F_t> * tallestMatrices,
        size_t nTallestMatrices,
        F_t(*digests)[HashParams::DIGEST_WIDTH],
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        merkle_tree_kernels::firstDigestLayer<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(tallestMatrices, nTallestMatrices,
                                            digests);
    }

    extern "C" void compress_and_inject(
        F_t(*prevLayer)[HashParams::DIGEST_WIDTH],
        size_t nPrevLayer,
        Matrix<F_t> * matricesToInject,
        size_t nMatricesToInject,
        F_t(*nextDigests)[HashParams::DIGEST_WIDTH],
        size_t nBlocks,
        size_t nThreadsPerBlock
    ) {
        merkle_tree_kernels::compressAndInject<HashParams>
            <<<nBlocks, nThreadsPerBlock>>>(prevLayer, nPrevLayer,
                                            matricesToInject, nMatricesToInject,
                                            nextDigests);
    }
}