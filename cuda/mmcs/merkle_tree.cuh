#pragma once

#include "../fields/bb31_t.cuh"
#include "../hashes/poseidon2/kernels.cuh"
#include "../matrix/matrix.cuh"

#include <stdio.h>

namespace merkle_tree_kernels {
using namespace poseidon2_bb31_16;

__global__ void firstDigestLayer(Matrix<bb31_t> *tallestMatrices,
                                 size_t nTallestMatrices,
                                 bb31_t (*digests)[poseidon2::DIGEST_WIDTH]) {
    int rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rowIdx >= tallestMatrices[0].height) {
        return;
    }

    Hasher hasher = newBabyBear16();

    HasherState state = HasherState();
    for (int i = 0; i < nTallestMatrices; i++) {
        hasher.absorbRow(&tallestMatrices[i], rowIdx, &state);
    }
    hasher.finalize(&state, digests[rowIdx]);
}

__global__ void compressAndInject(
    bb31_t (*prevLayer)[poseidon2::DIGEST_WIDTH], size_t nPrevLayer,
    Matrix<bb31_t> *matricesToInject, size_t nMatricesToInject,
    bb31_t (*nextDigests)[poseidon2::DIGEST_WIDTH]) {
    int rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rowIdx >= nPrevLayer / 2) {
        return;
    }

    Hasher hasher = newBabyBear16();

    if (nMatricesToInject == 0) {
        hasher.compress(prevLayer[rowIdx * 2], prevLayer[rowIdx * 2 + 1],
                        nextDigests[rowIdx]);
        return;
    }

    size_t nextLen = matricesToInject[0].height;
    // size_t nextLenPadded = nPrevLayer / 2;

    bb31_t defaultDigest[poseidon2::DIGEST_WIDTH] = {
        bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0)};

    bb31_t digest[poseidon2::DIGEST_WIDTH];
    hasher.compress(prevLayer[rowIdx * 2], prevLayer[rowIdx * 2 + 1], digest);

    if (rowIdx < nextLen) {
        bb31_t tallestDigest[poseidon2::DIGEST_WIDTH];
        HasherState state = HasherState();
        for (int i = 0; i < nMatricesToInject; i++) {
            hasher.absorbRow(&matricesToInject[i], rowIdx, &state);
        }
        hasher.finalize(&state, tallestDigest);
        hasher.compress(digest, tallestDigest, nextDigests[rowIdx]);
    } else {
        hasher.compress((bb31_t *)digest, (bb31_t *)defaultDigest,
                        (bb31_t *)nextDigests[rowIdx]);
    }
}

namespace column_major {}
}  // namespace merkle_tree_kernels

extern "C" namespace merkle_tree_gpu {
    using namespace poseidon2_bb31_16;

    extern "C" void first_digest_layer(
        Matrix<bb31_t> * tallestMatrices, size_t nTallestMatrices,
        bb31_t(*digests)[poseidon2::DIGEST_WIDTH], size_t nBlocks,
        size_t nThreadsPerBlock) {
        merkle_tree_kernels::firstDigestLayer<<<nBlocks, nThreadsPerBlock>>>(
            tallestMatrices, nTallestMatrices, digests);
    }

    extern "C" void compress_and_inject(
        bb31_t(*prevLayer)[poseidon2::DIGEST_WIDTH], size_t nPrevLayer,
        Matrix<bb31_t> * matricesToInject, size_t nMatricesToInject,
        bb31_t(*nextDigests)[poseidon2::DIGEST_WIDTH], size_t nBlocks,
        size_t nThreadsPerBlock) {
        merkle_tree_kernels::compressAndInject<<<nBlocks, nThreadsPerBlock>>>(
            prevLayer, nPrevLayer, matricesToInject, nMatricesToInject,
            nextDigests);
    }
}
