#include "../fields/bb31_t.cuh"
#include "../hashes/poseidon2_bb31_16/kernels.cuh"

#include <stdio.h>

struct RowMajorMatrix {
    DeviceSlice<bb31_t> data;
    int width;
    int height;
};

namespace merkle_tree_kernels {
using namespace poseidon2_bb31_16;

__global__ void firstDigestLayer(DeviceSlice<RowMajorMatrix> tallestMatrices,
                                 DeviceSlice<bb31_t[DIGEST_WIDTH]> digests,
                                 poseidon2_bb31_16::Hasher hasher) {
    int rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rowIdx >= tallestMatrices[0].height) {
        return;
    }

    poseidon2_bb31_16::HasherState state;
    for (int i = 0; i < tallestMatrices.length; i++) {
        DeviceSlice<bb31_t> row = tallestMatrices[i].data.slice(
            tallestMatrices[i].width * rowIdx, tallestMatrices[i].width);
        hasher.absorb(row, state);
    }
    hasher.finalize(state, digests[rowIdx]);
}

__global__ void compressAndInject(DeviceSlice<bb31_t[DIGEST_WIDTH]> prevLayer,
                                  DeviceSlice<RowMajorMatrix> matricesToInject,
                                  DeviceSlice<bb31_t[DIGEST_WIDTH]> nextDigests,
                                  poseidon2_bb31_16::Hasher hasher) {
    int rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rowIdx >= prevLayer.length / 2) {
        return;
    }

    if (matricesToInject.length == 0) {
        return;
    }

    size_t nextLen = matricesToInject[0].height;
    size_t nextLenPadded = prevLayer.length / 2;

    bb31_t defaultDigest[poseidon2_bb31_16::DIGEST_WIDTH] = {
        bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0), bb31_t(0)};

    bb31_t digest[poseidon2_bb31_16::DIGEST_WIDTH];
    hasher.compress(prevLayer[rowIdx * 2], prevLayer[rowIdx * 2 + 1], digest);

    if (rowIdx < nextLen) {
        bb31_t tallestDigest[poseidon2_bb31_16::DIGEST_WIDTH];
        poseidon2_bb31_16::HasherState state;
        for (int i = 0; i < matricesToInject.length; i++) {
            DeviceSlice<bb31_t> row = matricesToInject[i].data.slice(
                matricesToInject[i].width * rowIdx, matricesToInject[i].width);
            hasher.absorb(row, state);
        }
        hasher.finalize(state, tallestDigest);
        hasher.compress(digest, tallestDigest, nextDigests[rowIdx]);
    } else {
        hasher.compress((bb31_t*)digest, (bb31_t*)defaultDigest,
                        (bb31_t*)nextDigests[rowIdx]);
    }
}
}  // namespace merkle_tree_kernels

extern "C" namespace merkle_tree_gpu {
    using namespace poseidon2_bb31_16;

    extern "C" void firstDigestLayer(
        DeviceSlice<RowMajorMatrix> tallestMatrices,
        DeviceSlice<bb31_t[DIGEST_WIDTH]> digests,
        DeviceSlice<bb31_t[WIDTH]> external_rc, DeviceSlice<bb31_t> internal_rc,
        size_t nBlocks, size_t nThreadsPerBlock) {
        Hasher hasher = Hasher(external_rc, internal_rc);
        merkle_tree_kernels::firstDigestLayer<<<nBlocks, nThreadsPerBlock>>>(
            tallestMatrices, digests, hasher);
    }

    extern "C" void compressAndInject(
        DeviceSlice<bb31_t[DIGEST_WIDTH]> prevLayer,
        DeviceSlice<RowMajorMatrix> matricesToInject,
        DeviceSlice<bb31_t[DIGEST_WIDTH]> nextDigests,
        DeviceSlice<bb31_t[WIDTH]> external_rc, DeviceSlice<bb31_t> internal_rc,
        size_t nBlocks, size_t nThreadsPerBlock) {
        Hasher hasher = Hasher(external_rc, internal_rc);
        merkle_tree_kernels::compressAndInject<<<nBlocks, nThreadsPerBlock>>>(
            prevLayer, matricesToInject, nextDigests, hasher);
    }
}
