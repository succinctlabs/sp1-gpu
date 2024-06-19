#pragma once

#include "../../fields/bb31_t.cuh"
#include "../../matrix/matrix.cuh"
#include "../../utils/vector.cuh"

#include <stdio.h>

namespace poseidon2 {

constexpr int RATE = 8;
constexpr int DIGEST_WIDTH = 8;

template <typename F>
__device__ void mdsLightPermutation4x4(F state[4]) {
    F t01 = state[0] + state[1];
    F t23 = state[2] + state[3];
    F t0123 = t01 + t23;
    F t01123 = t0123 + state[1];
    F t01233 = t0123 + state[3];
    state[3] = t01233 + (state[0] << 1);
    state[1] = t01123 + (state[2] << 1);
    state[0] = t01123 + t01;
    state[2] = t01233 + t23;
}

template <typename F, int WIDTH>
__device__ void externalLinearLayer(F state[WIDTH]) {
    for (int i = 0; i < WIDTH; i += 4) {
        mdsLightPermutation4x4<F>(state + i);
    }

    F sums[4] = {state[0], state[1], state[2], state[3]};
    for (int i = 4; i < WIDTH; i += 4) {
        sums[0] += state[i];
        sums[1] += state[i + 1];
        sums[2] += state[i + 2];
        sums[3] += state[i + 3];
    }

    for (int i = 0; i < WIDTH; i++) {
        state[i] += sums[i % 4];
    }
}

template <typename F, int WIDTH>
__device__ void matmulInternal(F state[WIDTH], F matInternalDiagM1[WIDTH]) {
    F sum = F{0};
    for (int i = 0; i < WIDTH; i++) {
        sum += state[i];
    }

    for (int i = 0; i < WIDTH; i++) {
        state[i] *= matInternalDiagM1[i];
        state[i] += sum;
    }
}

template <typename F, int WIDTH>
__device__ void internalLinearLayer(F state[WIDTH], F matInternalDiagM1[WIDTH],
                                    F montyInverse) {
    matmulInternal<F, WIDTH>(state, matInternalDiagM1);
    for (int i = 0; i < WIDTH; i++) {
        state[i] = state[i] * montyInverse;
    }
}

template <typename F, int WIDTH>
__device__ void addRc(F state[WIDTH], const F *externalRoundConstants,
                      int round) {
    for (int i = 0; i < WIDTH; i++) {
        state[i] += externalRoundConstants[round * WIDTH + i];
    }
}

template <typename F, int WIDTH>
__device__ void sbox(F state[WIDTH], int D) {
    for (int i = 0; i < WIDTH; i++) {
        state[i] ^= D;
    }
}

template <typename F, int WIDTH>
struct HasherState {
    F data[WIDTH];
    size_t index;

    __device__ HasherState() {
        for (int i = 0; i < WIDTH; ++i) {
            data[i] = F(0);
        }
        index = 0;
    }
};

template <typename F, int WIDTH, int D, int ROUNDS_F, int ROUNDS_P>
class Hasher {
   private:
    F *internalRoundConstants;
    F *externalRoundConstants;
    F *matInternalDiagM1;
    F montyInverse;

   public:
    // Constructor
    Hasher() {
        cudaMalloc(&internalRoundConstants, ROUNDS_P * sizeof(F));
        cudaMalloc(&externalRoundConstants, ROUNDS_F * WIDTH * sizeof(F));
        cudaMalloc(&matInternalDiagM1, WIDTH * sizeof(F));
    }

    // Destructor
    ~Hasher() {
        cudaFree(internalRoundConstants);
        cudaFree(externalRoundConstants);
        cudaFree(matInternalDiagM1);
    }

    // Method to set constant arrays
    void setConstants(const F *internalRC, const F *externalRC,
                      const F *internalDiagM1, F inverse) {
        cudaMemcpy(internalRoundConstants, internalRC, ROUNDS_P * sizeof(F),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(externalRoundConstants, externalRC,
                   ROUNDS_F * WIDTH * sizeof(F), cudaMemcpyHostToDevice);
        cudaMemcpy(matInternalDiagM1, internalDiagM1, WIDTH * sizeof(F),
                   cudaMemcpyHostToDevice);
        montyInverse = inverse;
    }

    __device__ void permute(F in[WIDTH], F out[WIDTH]) {
        F state[WIDTH];
        for (int i = 0; i < WIDTH; i++) {
            state[i] = in[i];
        }

        externalLinearLayer<F, WIDTH>(state);

        int rounds_f_half = ROUNDS_F / 2;
        for (int i = 0; i < rounds_f_half; i++) {
            addRc<F, WIDTH>(state, externalRoundConstants, i);
            sbox<F, WIDTH>(state, D);
            externalLinearLayer<F, WIDTH>(state);
        }

        for (int i = 0; i < ROUNDS_P; i++) {
            state[0] += internalRoundConstants[i];
            state[0] ^= D;
            internalLinearLayer<F, WIDTH>(state, matInternalDiagM1,
                                          montyInverse);
        }

        for (int i = rounds_f_half; i < ROUNDS_F; i++) {
            addRc<F, WIDTH>(state, externalRoundConstants, i);
            sbox<F, WIDTH>(state, D);
            externalLinearLayer<F, WIDTH>(state);
        }

        for (int i = 0; i < WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ void compress(F left[DIGEST_WIDTH], F right[DIGEST_WIDTH],
                             F out[DIGEST_WIDTH]) {
        F state[WIDTH];
        for (int i = 0; i < DIGEST_WIDTH; i++) {
            state[i] = left[i];
            state[i + DIGEST_WIDTH] = right[i];
        }
        permute(state, state);
        for (int i = 0; i < DIGEST_WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ void hash(F *in, size_t nIn, F out[DIGEST_WIDTH]) {
        F state[WIDTH];
        for (int i = 0; i < WIDTH; i++) {
            state[i] = F(0);
        }

        for (int i = 0; i < nIn; i += RATE) {
            for (int j = 0; j < RATE; j++) {
                if (i + j < nIn) {
                    state[j] = in[i + j];
                }
            }
            permute(state, state);
        }

        for (int i = 0; i < DIGEST_WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ void absorb(F *in, size_t nIn, HasherState<F, WIDTH> *state) {
        for (int i = 0; i < nIn; i++) {
            state->data[state->index] = in[i];
            state->index++;
            if (state->index == RATE) {
                permute(state->data, state->data);
                state->index = 0;
            }
        }
    }

    __device__ void absorbRow(Matrix<F> *in, int row_idx,
                              HasherState<F, WIDTH> *state) {
        if (in->row_major) {
            F *row = &in->values[in->width * row_idx];
            absorb(row, in->width, state);
        } else {
            for (int j = 0; j < in->width; j++) {
                absorb(&in->values[j * in->height + row_idx], 1, state);
            }
        }
    }

    __device__ void finalize(HasherState<F, WIDTH> *state,
                             F out[DIGEST_WIDTH]) {
        if (state->index != 0) {
            permute(state->data, state->data);
        }
        for (int i = 0; i < DIGEST_WIDTH; i++) {
            out[i] = state->data[i];
        }
    }
};

}  // namespace poseidon2
