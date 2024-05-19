#include "constants.cuh"

#include "../../fields/bb31_t.cuh"
#include "../../utils/vector.cuh"

#include <stdio.h>

namespace poseidon2_bb31_16 {

constexpr int ROUNDS_F = 8;
constexpr int ROUNDS_P = 13;
constexpr int WIDTH = 16;
constexpr int DIGEST_WIDTH = 8;
constexpr int RATE = 8;
constexpr int D = 7;

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

__device__ void externalLinearLayer(bb31_t state[WIDTH]) {
    for (int i = 0; i < WIDTH; i += 4) {
        mdsLightPermutation4x4(state + i);
    }

    bb31_t sums[4] = {state[0], state[1], state[2], state[3]};
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

__device__ void matmulInternal(bb31_t state[WIDTH],
                               bb31_t matInternalDiagM1[WIDTH]) {
    bb31_t sum = bb31_t{0};
    for (int i = 0; i < WIDTH; i++) {
        sum += state[i];
    }

    for (int i = 0; i < WIDTH; i++) {
        state[i] *= matInternalDiagM1[i];
        state[i] += sum;
    }
}

__device__ void internalLinearLayer(bb31_t state[WIDTH]) {
    bb31_t montyInverse = bb31_t(943718400);
    bb31_t matInternalDiagM1[WIDTH] = {
        bb31_t(2013265919), bb31_t(1),    bb31_t(2),    bb31_t(4),
        bb31_t(8),          bb31_t(16),   bb31_t(32),   bb31_t(64),
        bb31_t(128),        bb31_t(256),  bb31_t(512),  bb31_t(1024),
        bb31_t(2048),       bb31_t(4096), bb31_t(8192), bb31_t(32768),
    };
    matmulInternal(state, matInternalDiagM1);
    for (int i = 0; i < WIDTH; i++) {
        state[i] = state[i] * montyInverse;
    }
}

__device__ void addRc(bb31_t state[WIDTH], bb31_t rc[WIDTH]) {
    for (int i = 0; i < WIDTH; i++) {
        state[i] += rc[i];
    }
}

__device__ void sbox(bb31_t state[WIDTH]) {
    for (int i = 0; i < WIDTH; i++) {
        state[i] ^= D;
    }
}

struct HasherState {
    bb31_t data[WIDTH];
    size_t index;

    __device__ HasherState() {
        for (int i = 0; i < WIDTH; ++i) {
            data[i] = bb31_t(0);
        }
        index = 0;
    }
};

class Hasher {
   public:
    __device__ void permute(bb31_t in[WIDTH], bb31_t out[WIDTH]) {
        bb31_t state[WIDTH];
        for (int i = 0; i < WIDTH; i++) {
            state[i] = in[i];
        }

        externalLinearLayer(state);

        int rounds_f_half = ROUNDS_F / 2;
        for (int i = 0; i < rounds_f_half; i++) {
            addRc(state, EXTERNAL_ROUND_CONSTANTS[i]);
            sbox(state);
            externalLinearLayer(state);
        }

        for (int i = 0; i < ROUNDS_P; i++) {
            state[0] += INTERNAL_ROUND_CONSTANTS[i];
            state[0] ^= D;
            internalLinearLayer(state);
        }

        for (int i = rounds_f_half; i < ROUNDS_F; i++) {
            addRc(state, EXTERNAL_ROUND_CONSTANTS[i]);
            sbox(state);
            externalLinearLayer(state);
        }

        for (int i = 0; i < WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ void compress(bb31_t left[DIGEST_WIDTH],
                             bb31_t right[DIGEST_WIDTH],
                             bb31_t out[DIGEST_WIDTH]) {
        bb31_t state[WIDTH];
        for (int i = 0; i < DIGEST_WIDTH; i++) {
            state[i] = left[i];
            state[i + DIGEST_WIDTH] = right[i];
        }
        permute(state, state);
        for (int i = 0; i < DIGEST_WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ void hash(bb31_t *in, size_t nIn, bb31_t out[DIGEST_WIDTH]) {
        bb31_t state[WIDTH];
        for (int i = 0; i < WIDTH; i++) {
            state[i] = bb31_t(0);
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

    __device__ void absorb(bb31_t *in, size_t nIn, HasherState *state) {
        for (int i = 0; i < nIn; i++) {
            state->data[state->index] = in[i];
            state->index++;
            if (state->index == RATE) {
                permute(state->data, state->data);
                state->index = 0;
            }
        }
    }

    __device__ void finalize(HasherState *state, bb31_t out[DIGEST_WIDTH]) {
        if (state->index != 0) {
            permute(state->data, state->data);
        }
        for (int i = 0; i < DIGEST_WIDTH; i++) {
            out[i] = state->data[i];
        }
    }
};

}  // namespace poseidon2_bb31_16
