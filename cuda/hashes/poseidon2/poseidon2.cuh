#pragma once

#include "../../matrix/matrix.cuh"
#include "../../utils/vector.cuh"

namespace poseidon2 {

template<typename F, int WIDTH>
__device__ void addExtRc(F state[WIDTH], const F rc[WIDTH]) {
    for (int i = 0; i < WIDTH; i++) {
        state[i] += rc[i];
    }
}

template<typename F, int WIDTH>
__device__ void sbox(F state[WIDTH], const int D) {
    for (int i = 0; i < WIDTH; i++) {
        state[i] ^= D;
    }
}

template<typename F>
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

template<typename F, int WIDTH>
__device__ void externalLinearLayer(F state[WIDTH]) {
    switch (WIDTH) {
        case 2: {
            F sum = state[0] + state[1];
            state[0] += sum;
            state[1] += sum;
            break;
        }
        case 3: {
            F sum = state[0] + state[1] + state[2];
            state[0] += sum;
            state[1] += sum;
            state[2] += sum;
            break;
        }
        case 4:
            mdsLightPermutation4x4<F>(state);
            break;
        case 8:
        case 12:
        case 16:
        case 20:
        case 24:
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

            break;
    }
}

template<typename F, int WIDTH>
__device__ void
matmulInternal(F state[WIDTH], const F matInternalDiagM1[WIDTH]) {
    F sum = F {0};
    for (int i = 0; i < WIDTH; i++) {
        sum += state[i];
    }

    for (int i = 0; i < WIDTH; i++) {
        state[i] *= matInternalDiagM1[i];
        state[i] += sum;
    }
}

template<typename F, int WIDTH>
__device__ void internalLinearLayer(
    F state[WIDTH],
    const F matInternalDiagM1[WIDTH],
    const F montyInverse
) {
    switch (WIDTH) {
        case 2: {
            // [2, 1]
            // [1, 3]
            F s = state[0] + state[1];
            state[0] += s;
            state[1] *= 2;
            state[1] += s;
            break;
        }
        case 3: {
            // [2, 1, 1]
            // [1, 2, 1]
            // [1, 1, 3]
            F s = state[0] + state[1] + state[2];
            state[0] += s;
            state[1] += s;
            state[2] *= 2;
            state[2] *= s;
            break;
        }
        case 4:
        case 8:
        case 12:
        case 16:
        case 20:
        case 24:
            matmulInternal<F, WIDTH>(state, matInternalDiagM1);
            for (int i = 0; i < WIDTH; i++) {
                state[i] = state[i] * montyInverse;  // ?
            }
            break;
    }
}

template<typename Params>
struct HasherState {
    using F = typename Params::F;

    F data[Params::WIDTH];
    size_t index;

    __device__ HasherState() : index(0) {
        for (int i = 0; i < Params::WIDTH; ++i) {
            data[i] = F(0);
        }
    }
};

template<typename Params>
class Hasher {
    using F = typename Params::F;

  public:
    __device__ void permute(F in[Params::WIDTH], F out[Params::WIDTH]) {
        F state[Params::WIDTH];
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i] = in[i];
        }

        externalLinearLayer<F, Params::WIDTH>(state);

        int rounds_f_half = Params::ROUNDS_F / 2;
        for (int i = 0; i < rounds_f_half; i++) {
            addExtRc<F, Params::WIDTH>(
                state,
                Params::EXTERNAL_ROUND_CONSTANTS[i]
            );
            sbox<F, Params::WIDTH>(state, Params::D);
            externalLinearLayer<F, Params::WIDTH>(state);
        }

        for (int i = 0; i < Params::ROUNDS_P; i++) {
            state[0] += Params::INTERNAL_ROUND_CONSTANTS[i];
            state[0] ^= Params::D;
            internalLinearLayer<F, Params::WIDTH>(
                state,
                Params::MAT_INTERNAL_DIAG_M1,
                Params::MONTY_INVERSE
            );
        }

        for (int i = rounds_f_half; i < Params::ROUNDS_F; i++) {
            addExtRc<F, Params::WIDTH>(
                state,
                Params::EXTERNAL_ROUND_CONSTANTS[i]
            );
            sbox<F, Params::WIDTH>(state, Params::D);
            externalLinearLayer<F, Params::WIDTH>(state);
        }

        for (int i = 0; i < Params::WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ void compress(
        F left[Params::DIGEST_WIDTH],
        F right[Params::DIGEST_WIDTH],
        F out[Params::DIGEST_WIDTH]
    ) {
        F state[Params::WIDTH];
        for (int i = 0; i < Params::DIGEST_WIDTH; i++) {
            state[i] = left[i];
            state[i + Params::DIGEST_WIDTH] = right[i];
        }
        permute(state, state);
        for (int i = 0; i < Params::DIGEST_WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ void hash(F* in, size_t nIn, F out[Params::DIGEST_WIDTH]) {
        F state[Params::WIDTH];
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i] = F(0);
        }

        for (int i = 0; i < nIn; i += Params::RATE) {
            for (int j = 0; j < Params::RATE; j++) {
                if (i + j < nIn) {
                    state[j] = in[i + j];
                }
            }
            permute(state, state);
        }

        for (int i = 0; i < Params::DIGEST_WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ void absorb(F* in, size_t nIn, HasherState<Params>* state) {
        for (int i = 0; i < nIn; i++) {
            state->data[state->index] = in[i];
            state->index++;
            if (state->index == Params::RATE) {
                permute(state->data, state->data);
                state->index = 0;
            }
        }
    }

    __device__ void
    absorbRow(Matrix<F>* in, int row_idx, HasherState<Params>* state) {
        if (in->row_major) {
            F* row = &in->values[in->width * row_idx];
            absorb(row, in->width, state);
        } else {
            for (int j = 0; j < in->width; j++) {
                absorb(&in->values[j * in->height + row_idx], 1, state);
            }
        }
    }

    __device__ void
    finalize(HasherState<Params>* state, F out[Params::DIGEST_WIDTH]) {
        if (state->index != 0) {
            permute(state->data, state->data);
        }
        for (int i = 0; i < Params::DIGEST_WIDTH; i++) {
            out[i] = state->data[i];
        }
    }
};

}  // namespace poseidon2
