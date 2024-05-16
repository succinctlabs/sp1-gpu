#include "kernels.cuh"

extern "C" namespace poseidon2_bb31_16_gpu {
    extern "C" void permute(
        DeviceSlice<bb31_t[poseidon2_bb31_16::WIDTH]> in,
        DeviceSlice<bb31_t[poseidon2_bb31_16::WIDTH]> out,
        DeviceSlice<bb31_t[poseidon2_bb31_16::WIDTH]> external_rc,
        DeviceSlice<bb31_t> internal_rc, size_t n, size_t nBlocks,
        size_t nThreadsPerBlock) {
        poseidon2_bb31_16::Hasher hasher =
            poseidon2_bb31_16::Hasher(external_rc, internal_rc);
        poseidon2_bb31_16_kernels::permute<<<nBlocks, nThreadsPerBlock>>>(
            hasher, in, out, n);
    }
}  // namespace poseidon2_gpu