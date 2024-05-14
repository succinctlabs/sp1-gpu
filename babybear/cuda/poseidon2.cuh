#include "bb31_t.cuh"
#include <stdio.h>

#define ROUNDS_F 8
#define ROUNDS_P 13
#define WIDTH 16
#define D 7

__device__ void mdsLightPermutation4x4(bb31_t state[4]) {
    bb31_t t01 = state[0] + state[1];
    bb31_t t23 = state[2] + state[3];
    bb31_t t0123 = t01 + t23;
    bb31_t t01123 = t0123 + state[1];
    bb31_t t01233 = t0123 + state[3];
    state[3] = t01233 + (state[0] << 1);
    state[1] = t01123 + (state[2] << 1);
    state[0] = t01123 + t01;
    state[2] = t01233 + t23;
}

__device__ void externalLinearLayer(bb31_t state[WIDTH])
{
    for (int i = 0; i < WIDTH; i += 4)
    {
        mdsLightPermutation4x4(state + i);
    }

    bb31_t sums[4] = {state[0], state[1], state[2], state[3]};
#pragma unroll
    for (int i = 4; i < WIDTH; i += 4)
    {
        sums[0] += state[i];
        sums[1] += state[i + 1];
        sums[2] += state[i + 2];
        sums[3] += state[i + 3];
    }

#pragma unroll
    for (int i = 0; i < WIDTH; i++)
    {
        state[i] += sums[i % 4];
    }
}

__global__ void testExternalLinearLayer(bb31_t state[WIDTH], int n)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    externalLinearLayer(state + idx * WIDTH);
}

__device__ void matmulInternal(bb31_t state[WIDTH], bb31_t matInternalDiagM1[WIDTH])
{
    bb31_t sum = bb31_t{0};
#pragma unroll
    for (int i = 0; i < WIDTH; i++)
    {
        sum += state[i];
    }
#pragma unroll
    for (int i = 0; i < WIDTH; i++)
    {
        state[i] *= matInternalDiagM1[i];
        state[i] += sum;
    }
}

__device__ void internalLinearLayer(bb31_t state[WIDTH])
{
    bb31_t montyInverse = bb31_t(943718400);
    bb31_t matInternalDiagM1[WIDTH] = {
        bb31_t(2013265919),
        bb31_t(1),
        bb31_t(2),
        bb31_t(4),
        bb31_t(8),
        bb31_t(16),
        bb31_t(32),
        bb31_t(64),
        bb31_t(128),
        bb31_t(256),
        bb31_t(512),
        bb31_t(1024),
        bb31_t(2048),
        bb31_t(4096),
        bb31_t(8192),
        bb31_t(32768),
    };
    matmulInternal(state, matInternalDiagM1);
#pragma unroll
    for (int i = 0; i < WIDTH; i++)
    {
        state[i] = state[i] * montyInverse;
    }
}

__global__ void testInternalLinearLayer(bb31_t state[WIDTH], int n)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    internalLinearLayer(state + idx * WIDTH);
}

__device__ void addRc(bb31_t state[WIDTH], bb31_t rc[WIDTH])
{
#pragma unroll
    for (int i = 0; i < WIDTH; i++)
    {
        state[i] += rc[i];
    }
}

__device__ void sbox(bb31_t state[WIDTH])
{
#pragma unroll
    for (int i = 0; i < WIDTH; i++)
    {
        state[i] ^= D;
    }
}

__global__ void poseidon2PermuteKernel(bb31_t *in, bb31_t *out, bb31_t *external_rc, bb31_t *internal_rc, int n)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n)
    {
        return;
    }

    bb31_t state[WIDTH];
#pragma unroll
    for (int i = 0; i < WIDTH; i++)
    {
        state[i] = in[idx * WIDTH + i];
    }

    externalLinearLayer(state);

    int rounds_f_half = ROUNDS_F / 2;
#pragma unroll
    for (int i = 0; i < rounds_f_half; i++)
    {
        addRc(state, external_rc + i * WIDTH);
        sbox(state);
        externalLinearLayer(state);
    }

#pragma unroll
    for (int i = 0; i < ROUNDS_P; i++)
    {
        state[0] += internal_rc[i];
        state[0] ^= D;
        internalLinearLayer(state);
    }

#pragma unroll
    for (int i = rounds_f_half; i < ROUNDS_F; i++)
    {
        addRc(state, external_rc + i * WIDTH);
        sbox(state);
        externalLinearLayer(state);
    }

#pragma unroll
    for (int i = 0; i < WIDTH; i++)
    {
        out[idx * WIDTH + i] = state[i];
    }
}
