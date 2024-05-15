#include "poseidon2.cuh"

__global__ void vectorExternalLinearLayer(bb31_t state[WIDTH], int n) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    externalLinearLayer(state + idx * WIDTH);
}

__global__ void vectorInternalLinearLayer(bb31_t state[WIDTH], int n) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    internalLinearLayer(state + idx * WIDTH);
}

__global__ void vectorPoseidon2Permute(Poseidon2Device hasher, bb31_t *in,
                                       bb31_t *out, int n) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    hasher.permute(in + idx * WIDTH, out + idx * WIDTH);
}

__global__ void vectorPoseidon2Hash(Poseidon2Device hasher, bb31_t *in, int len,
                                    bb31_t *out, int n) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    hasher.hash(in + idx * len, len, out + idx * DIGEST_WIDTH);
}