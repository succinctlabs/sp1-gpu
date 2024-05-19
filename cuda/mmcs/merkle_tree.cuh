#include "../fields/bb31_t.cuh"
#include "../hashes/poseidon2_bb31_16/kernels.cuh"
#include "../utils/matrix.cuh"

#include <stdio.h>

namespace merkle_tree_kernels {
using namespace poseidon2_bb31_16;

__global__ void firstDigestLayer(RowMajorMatrix *tallestMatrices,
                                 size_t nTallestMatrices,
                                 bb31_t (*digests)[DIGEST_WIDTH],
                                 poseidon2_bb31_16::Hasher hasher) {
    int rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rowIdx >= tallestMatrices[0].height) {
        return;
    }

    poseidon2_bb31_16::HasherState state;
    for (int i = 0; i < nTallestMatrices; i++) {
        bb31_t *row =
            tallestMatrices[i].values + tallestMatrices[i].width * rowIdx;
        hasher.absorb(row, tallestMatrices[i].width, state);
    }
    hasher.finalize(state, digests[rowIdx]);
}

__global__ void compressAndInject(bb31_t (*prevLayer)[DIGEST_WIDTH],
                                  size_t nPrevLayer,
                                  RowMajorMatrix *matricesToInject,
                                  size_t nMatricesToInject,
                                  bb31_t (*nextDigests)[DIGEST_WIDTH],
                                  poseidon2_bb31_16::Hasher hasher) {
    int rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rowIdx >= nPrevLayer / 2) {
        return;
    }

    if (nMatricesToInject == 0) {
        return;
    }

    size_t nextLen = matricesToInject[0].height;
    size_t nextLenPadded = nPrevLayer / 2;

    bb31_t defaultDigest[poseidon2_bb31_16::DIGEST_WIDTH] = {
        bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0)};

    bb31_t digest[poseidon2_bb31_16::DIGEST_WIDTH];
    hasher.compress(prevLayer[rowIdx * 2], prevLayer[rowIdx * 2 + 1], digest);

    if (rowIdx < nextLen) {
        bb31_t tallestDigest[poseidon2_bb31_16::DIGEST_WIDTH];
        poseidon2_bb31_16::HasherState state;
        for (int i = 0; i < nMatricesToInject; i++) {
            bb31_t *row =
                matricesToInject[i].values + matricesToInject[i].width * rowIdx;
            hasher.absorb(row, matricesToInject[i].width, state);
        }
        hasher.finalize(state, tallestDigest);
        hasher.compress(digest, tallestDigest, nextDigests[rowIdx]);
    } else {
        hasher.compress((bb31_t *)digest, (bb31_t *)defaultDigest,
                        (bb31_t *)nextDigests[rowIdx]);
    }
}
}  // namespace merkle_tree_kernels

extern "C" namespace merkle_tree_gpu {
    using namespace poseidon2_bb31_16;

    extern "C" void firstDigestLayer(RowMajorMatrix * tallestMatrices,
                                     size_t nTallestMatrices,
                                     bb31_t(*digests)[DIGEST_WIDTH],
                                     size_t nBlocks, size_t nThreadsPerBlock) {
        Hasher hasher = Hasher();
        merkle_tree_kernels::firstDigestLayer<<<nBlocks, nThreadsPerBlock>>>(
            tallestMatrices, nTallestMatrices, digests, hasher);
    }

    extern "C" void compressAndInject(
        bb31_t(*prevLayer)[DIGEST_WIDTH], size_t nPrevLayer,
        RowMajorMatrix * matricesToInject, size_t nMatricesToInject,
        bb31_t(*nextDigests)[DIGEST_WIDTH], size_t nBlocks,
        size_t nThreadsPerBlock) {
        Hasher hasher = Hasher();
        merkle_tree_kernels::compressAndInject<<<nBlocks, nThreadsPerBlock>>>(
            prevLayer, nPrevLayer, matricesToInject, nMatricesToInject,
            nextDigests, hasher);
    }
}
