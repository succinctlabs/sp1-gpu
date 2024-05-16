#include "kernels.cuh"

extern "C" namespace poseidon2_bb31_16_gpu {
    using namespace poseidon2_bb31_16;

    extern "C" void permute(
        DeviceSlice<bb31_t[WIDTH]> in, DeviceSlice<bb31_t[WIDTH]> out,
        DeviceSlice<bb31_t[WIDTH]> external_rc, DeviceSlice<bb31_t> internal_rc,
        size_t n, size_t nBlocks, size_t nThreadsPerBlock) {
        Hasher hasher = Hasher(external_rc, internal_rc);
        poseidon2_bb31_16_kernels::permute<<<nBlocks, nThreadsPerBlock>>>(
            hasher, in, out, n);
    }

    extern "C" void compress(DeviceSlice<bb31_t[DIGEST_WIDTH]> left,
                             DeviceSlice<bb31_t[DIGEST_WIDTH]> right,
                             DeviceSlice<bb31_t[DIGEST_WIDTH]> out,
                             DeviceSlice<bb31_t[WIDTH]> external_rc,
                             DeviceSlice<bb31_t> internal_rc, size_t n,
                             size_t nBlocks, size_t nThreadsPerBlock) {
        Hasher hasher = Hasher(external_rc, internal_rc);
        poseidon2_bb31_16_kernels::compress<<<nBlocks, nThreadsPerBlock>>>(
            hasher, left, right, out, n);
    }

    extern "C" void hash(
        DeviceSlice<bb31_t> in, int nIn, DeviceSlice<bb31_t[DIGEST_WIDTH]> out,
        DeviceSlice<bb31_t[WIDTH]> external_rc, DeviceSlice<bb31_t> internal_rc,
        size_t n, size_t nBlocks, size_t nThreadsPerBlock) {
        Hasher hasher = Hasher(external_rc, internal_rc);
        poseidon2_bb31_16_kernels::hash<<<nBlocks, nThreadsPerBlock>>>(
            hasher, in, nIn, out, n);
    }
}  // namespace poseidon2_gpu