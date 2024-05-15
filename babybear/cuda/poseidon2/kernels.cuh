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

__global__ void vectorPoseidon2Permute(bb31_t *in, bb31_t *out,
                                       bb31_t *external_rc, bb31_t *internal_rc,
                                       int n) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    Poseidon2 hasher = Poseidon2(external_rc, internal_rc);
    hasher.permute(in + idx * WIDTH, out + idx * WIDTH);
}

__global__ void vectorPoseidon2Hash(bb31_t *in, int len, bb31_t *out,
                                    bb31_t *external_rc, bb31_t *internal_rc,
                                    int n) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }
    Poseidon2 hasher = Poseidon2(external_rc, internal_rc);
    hasher.hash(in + idx * len, len, out + idx * DIGEST_WIDTH);
}