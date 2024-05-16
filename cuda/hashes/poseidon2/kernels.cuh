#include "poseidon2.cuh"

namespace poseidon2_kernels {
__global__ void permute(poseidon2::Hasher hasher,
                        DeviceSlice<bb31_t[poseidon2::WIDTH]> in,
                        DeviceSlice<bb31_t[poseidon2::WIDTH]> out, size_t n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }

    hasher.permute(in[idx], out[idx]);
}

__global__ void compress(poseidon2::Hasher hasher,
                         DeviceSlice<bb31_t[poseidon2::DIGEST_WIDTH]> left,
                         DeviceSlice<bb31_t[poseidon2::DIGEST_WIDTH]> right,
                         DeviceSlice<bb31_t[poseidon2::DIGEST_WIDTH]> out,
                         size_t n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }

    hasher.compress(left[idx], right[idx], out[idx]);
}

__global__ void hash(poseidon2::Hasher hasher, DeviceSlice<bb31_t> in, int nIn,
                     DeviceSlice<bb31_t> out, int n) {
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }

    hasher.hash(in.slice(idx * nIn, nIn), out.items + idx * poseidon2::WIDTH);
}
}  // namespace poseidon2_kernels