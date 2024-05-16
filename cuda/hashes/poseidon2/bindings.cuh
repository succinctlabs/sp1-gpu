#include "kernels.cuh"

extern "C" namespace poseidon2_gpu {
    extern "C" void permute(DeviceSlice<bb31_t[poseidon2::WIDTH]> in,
                            DeviceSlice<bb31_t[poseidon2::WIDTH]> out,
                            DeviceSlice<bb31_t[poseidon2::WIDTH]> external_rc,
                            DeviceSlice<bb31_t> internal_rc, size_t n,
                            size_t nBlocks, size_t nThreadsPerBlock) {
        poseidon2::Hasher hasher = poseidon2::Hasher(external_rc, internal_rc);
        poseidon2_kernels::permute<<<nBlocks, nThreadsPerBlock>>>(hasher, in,
                                                                  out, n);
    }
}  // namespace poseidon2_gpu