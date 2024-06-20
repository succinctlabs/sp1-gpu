#pragma once

#include "../../matrix/matrix.cuh"
#include "../../utils/vector.cuh"

#include <stdio.h>

namespace poseidon2 {

template <typename F, int WIDTH>
__device__ void addExtRc(F state[WIDTH], const F rc[WIDTH]) {
    for (int i = 0; i < WIDTH; i++) {
        state[i] += rc[i];
    }
}

template <typename F, int WIDTH>
__device__ void sbox(F state[WIDTH], const int D) {
    for (int i = 0; i < WIDTH; i++) {
        state[i] ^= D;
    }
}

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
__device__ void addIntRc(F state[WIDTH], const F rc[WIDTH], int round) {
    state[0] += rc[round];
}

template <typename F, int WIDTH>
__device__ void matmulInternal(F state[WIDTH],
                               const F matInternalDiagM1[WIDTH]) {
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
__device__ void internalLinearLayer(F state[WIDTH],
                                    const F matInternalDiagM1[WIDTH],
                                    const F montyInverse) {
    matmulInternal<F, WIDTH>(state, matInternalDiagM1);
    for (int i = 0; i < WIDTH; i++) {
        state[i] = state[i] * montyInverse;
    }
}

template <typename Params>
struct HasherState {
    using F = typename Params::F;
    static const size_t WIDTH = Params::WIDTH;

    F data[WIDTH];
    size_t index;

    __device__ HasherState() : index(0) {
        for (int i = 0; i < WIDTH; ++i) {
            data[i] = F(0);
        }
    }
};

template <typename Params>
class Hasher {
    using F = typename Params::F;
    static const int DIGEST_WIDTH = Params::DIGEST_WIDTH;
    static const int RATE = Params::RATE;
    static const int WIDTH = Params::WIDTH;
    static const int ROUNDS_F = Params::ROUNDS_F;
    static const int ROUNDS_P = Params::ROUNDS_P;

   public:
    __device__ void permute(F in[WIDTH], F out[WIDTH]) {
        F state[WIDTH];
        for (int i = 0; i < WIDTH; i++) {
            state[i] = in[i];
        }

        externalLinearLayer<F, WIDTH>(state);

        int rounds_f_half = ROUNDS_F / 2;
        for (int i = 0; i < rounds_f_half; i++) {
            addExtRc<F, WIDTH>(state, Params::getExternalRoundConstants()[i]);
            sbox<F, WIDTH>(state, Params::getD());
            externalLinearLayer<F, WIDTH>(state);
        }

        for (int i = 0; i < ROUNDS_P; i++) {
            addIntRc<F, WIDTH>(state, Params::getInternalRoundConstants(), i);
            state[0] ^= Params::getD();
            internalLinearLayer<F, WIDTH>(state, Params::getMatInternalDiagM1(),
                                          Params::getMontyInverse());
        }

        for (int i = rounds_f_half; i < ROUNDS_F; i++) {
            addExtRc<F, WIDTH>(state, Params::getExternalRoundConstants()[i]);
            sbox<F, WIDTH>(state, Params::getD());
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

    __device__ void absorb(F *in, size_t nIn, HasherState<Params> *state) {
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
                              HasherState<Params> *state) {
        if (in->row_major) {
            F *row = &in->values[in->width * row_idx];
            absorb(row, in->width, state);
        } else {
            for (int j = 0; j < in->width; j++) {
                absorb(&in->values[j * in->height + row_idx], 1, state);
            }
        }
    }

    __device__ void finalize(HasherState<Params> *state, F out[DIGEST_WIDTH]) {
        if (state->index != 0) {
            permute(state->data, state->data);
        }
        for (int i = 0; i < DIGEST_WIDTH; i++) {
            out[i] = state->data[i];
        }
    }
};

}  // namespace poseidon2
