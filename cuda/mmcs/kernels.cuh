#include "merkle_tree.cuh"

extern "C" namespace mmcs_gpu {
    using namespace poseidon2_bb31_16;

    extern "C" void firstDigestLayer(
        DeviceSlice<RowMajorMatrix> tallestMatrices,
        DeviceSlice<bb31_t[DIGEST_WIDTH]> digests,
        DeviceSlice<bb31_t[WIDTH]> external_rc, DeviceSlice<bb31_t> internal_rc,
        size_t n, size_t nBlocks, size_t nThreadsPerBlock) {
        mmcs_kernels::firstDigestLayer<<<nBlocks, nThreadsPerBlock>>>(
            tallestMatrices, digests, external_rc, internal_rc);
    }

    extern "C" void compressAndInject(
        DeviceSlice<bb31_t[DIGEST_WIDTH]> prevLayer,
        DeviceSlice<RowMajorMatrix> matricesToInject,
        DeviceSlice<bb31_t[DIGEST_WIDTH]> nextDigests,
        DeviceSlice<bb31_t[WIDTH]> external_rc, DeviceSlice<bb31_t> internal_rc,
        size_t n, size_t nBlocks, size_t nThreadsPerBlock) {
        mmcs_kernels::compressAndInject<<<nBlocks, nThreadsPerBlock>>>(
            prevLayer, matricesToInject, nextDigests, external_rc, internal_rc);
    }
}

namespace mmcs_kernels {
using namespace poseidon2_bb31_16;

__global__ void firstDigestLayer(DeviceSlice<RowMajorMatrix> tallestMatrices,
                                 DeviceSlice<bb31_t[DIGEST_WIDTH]> digests,
                                 DeviceSlice<bb31_t[WIDTH]> external_rc,
                                 DeviceSlice<bb31_t> internal_rc) {
    int rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rowIdx >= tallestMatrices[0].height) {
        return;
    }

    Hasher hasher = Hasher(external_rc, internal_rc);
    mmcs::FieldMerkleTree fmt = mmcs::FieldMerkleTree(hasher);
    fmt.firstDigestLayer(tallestMatrices, digests, rowIdx);
}

__global__ void compressAndInject(DeviceSlice<bb31_t[DIGEST_WIDTH]> prevLayer,
                                  DeviceSlice<RowMajorMatrix> matricesToInject,
                                  DeviceSlice<bb31_t[DIGEST_WIDTH]> nextDigests,
                                  DeviceSlice<bb31_t[WIDTH]> external_rc,
                                  DeviceSlice<bb31_t> internal_rc) {
    int rowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (rowIdx >= prevLayer.length / 2) {
        return;
    }

    Hasher hasher = Hasher(external_rc, internal_rc);
    mmcs::FieldMerkleTree fmt = mmcs::FieldMerkleTree(hasher);
    fmt.compressAndInject(prevLayer, matricesToInject, nextDigests, rowIdx);
}
}  // namespace mmcs_kernels