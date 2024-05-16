#include "poseidon2_bb31_16.cuh"

namespace poseidon2_bb31_16_kernels {
__global__ void permute(poseidon2_bb31_16::Hasher hasher,
                        DeviceSlice<bb31_t[poseidon2_bb31_16::WIDTH]> in,
                        DeviceSlice<bb31_t[poseidon2_bb31_16::WIDTH]> out,
                        size_t n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }

    hasher.permute(in[idx], out[idx]);
}

__global__ void compress(
    poseidon2_bb31_16::Hasher hasher,
    DeviceSlice<bb31_t[poseidon2_bb31_16::DIGEST_WIDTH]> left,
    DeviceSlice<bb31_t[poseidon2_bb31_16::DIGEST_WIDTH]> right,
    DeviceSlice<bb31_t[poseidon2_bb31_16::DIGEST_WIDTH]> out, size_t n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }

    hasher.compress((bb31_t*)(left.items + idx), (bb31_t*)(right.items + idx),
                    (bb31_t*)(out.items + idx));
}

__global__ void hash(poseidon2_bb31_16::Hasher hasher, DeviceSlice<bb31_t> in,
                     int nIn,
                     DeviceSlice<bb31_t[poseidon2_bb31_16::DIGEST_WIDTH]> out,
                     int n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    hasher.hash(in.slice(idx * nIn, nIn), (bb31_t*)(out.items + idx));
}
}  // namespace poseidon2_bb31_16_kernels