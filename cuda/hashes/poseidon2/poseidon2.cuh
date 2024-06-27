#pragma once

#include "../../matrix/matrix.cuh"
#include "../../utils/vector.cuh"

namespace poseidon2 {

template<typename Params>
struct HasherState {
    using F = typename Params::F;

    F data[Params::WIDTH];
    size_t index;

    __device__ HasherState() : index(0) {
        for (int i = 0; i < Params::WIDTH; ++i) {
            data[i].zero();
        }
    }
};

template<typename Params>
class Hasher {
    using F = typename Params::F;
    using pF = typename Params::pF;

  private:
    __device__ void addExtRc(F state[Params::WIDTH], pF rc) {
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i] += rc[i];
        }
    }

    __device__ void sbox(F state[Params::WIDTH]) {
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i] ^= Params::D;
        }
    }

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

    __device__ void externalLinearLayer(F state[Params::WIDTH]) {
        switch (Params::WIDTH) {
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
                mdsLightPermutation4x4(state);
                break;
            case 8:
            case 12:
            case 16:
            case 20:
            case 24:
                for (int i = 0; i < Params::WIDTH; i += 4) {
                    mdsLightPermutation4x4(state + i);
                }

                F sums[4] = {state[0], state[1], state[2], state[3]};
                for (int i = 4; i < Params::WIDTH; i += 4) {
                    sums[0] += state[i];
                    sums[1] += state[i + 1];
                    sums[2] += state[i + 2];
                    sums[3] += state[i + 3];
                }

                for (int i = 0; i < Params::WIDTH; i++) {
                    state[i] += sums[i % 4];
                }

                break;
        }
    }

    __device__ void
    matmulInternal(F state[Params::WIDTH], pF matInternalDiagM1) {
        F sum;
        sum.zero();
        for (int i = 0; i < Params::WIDTH; i++) {
            sum += state[i];
        }

        for (int i = 0; i < Params::WIDTH; i++) {
            state[i] *= matInternalDiagM1[i];
            state[i] += sum;
        }
    }

    __device__ void internalLinearLayer(
        F state[Params::WIDTH],
        pF matInternalDiagM1,
        F montyInverse
    ) {
        switch (Params::WIDTH) {
            case 2: {
                // [2, 1]
                // [1, 3]
                F s = state[0] + state[1];
                state[0] += s;
                // state[1] *= 2;
                state[1] += state[1];
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
                // state[2] *= 2;
                state[2] += state[2];
                state[2] *= s;
                break;
            }
            case 4:
            case 8:
            case 12:
            case 16:
            case 20:
            case 24:
                matmulInternal(state, matInternalDiagM1);
                for (int i = 0; i < Params::WIDTH; i++) {
                    state[i] = state[i] * montyInverse;
                }
                break;
        }
    }

  public:
    // TODO: are we sacrificing infornation about the length of the params?
    // TODO: poseidon2 params should be passed around more cleanly
    __device__ void permute(
        F in[Params::WIDTH],
        F out[Params::WIDTH],
        pF internalRoundConstants,
        pF externalRoundConstants,
        pF matInternalDiagM1,
        F montyInverse
    ) {
        F state[Params::WIDTH];
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i] = in[i];
        }

        externalLinearLayer(state);

        int rounds_f_half = Params::ROUNDS_F / 2;
        for (int i = 0; i < rounds_f_half; i++) {
            addExtRc(state, externalRoundConstants + i * Params::WIDTH);
            sbox(state);
            externalLinearLayer(state);
        }

        for (int i = 0; i < Params::ROUNDS_P; i++) {
            state[0] += internalRoundConstants[i];
            state[0] ^= Params::D;
            internalLinearLayer(state, matInternalDiagM1, montyInverse);
        }

        for (int i = rounds_f_half; i < Params::ROUNDS_F; i++) {
            addExtRc(state, externalRoundConstants + i * Params::WIDTH);
            sbox(state);
            externalLinearLayer(state);
        }

        for (int i = 0; i < Params::WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ void compress(
        F left[Params::DIGEST_WIDTH],
        F right[Params::DIGEST_WIDTH],
        F out[Params::DIGEST_WIDTH],
        pF internalRoundConstants,
        pF externalRoundConstants,
        pF matInternalDiagM1,
        F montyInverse
    ) {
        F state[Params::WIDTH];
        for (int i = 0; i < Params::DIGEST_WIDTH; i++) {
            state[i] = left[i];
            state[i + Params::DIGEST_WIDTH] = right[i];
        }
        permute(
            state,
            state,
            internalRoundConstants,
            externalRoundConstants,
            matInternalDiagM1,
            montyInverse
        );
        for (int i = 0; i < Params::DIGEST_WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ void hash(
        F* in,
        size_t nIn,
        F out[Params::DIGEST_WIDTH],
        pF internalRoundConstants,
        pF externalRoundConstants,
        pF matInternalDiagM1,
        F montyInverse
    ) {
        F state[Params::WIDTH];
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i].zero();
        }

        for (int i = 0; i < nIn; i += Params::RATE) {
            for (int j = 0; j < Params::RATE; j++) {
                if (i + j < nIn) {
                    state[j] = in[i + j];
                }
            }
            permute(
                state,
                state,
                internalRoundConstants,
                externalRoundConstants,
                matInternalDiagM1,
                montyInverse
            );
        }

        for (int i = 0; i < Params::DIGEST_WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ void absorb(
        F* in,
        size_t nIn,
        HasherState<Params>* state,
        pF internalRoundConstants,
        pF externalRoundConstants,
        pF matInternalDiagM1,
        F montyInverse
    ) {
        for (int i = 0; i < nIn; i++) {
            state->data[state->index] = in[i];
            state->index++;
            if (state->index == Params::RATE) {
                permute(
                    state->data,
                    state->data,
                    internalRoundConstants,
                    externalRoundConstants,
                    matInternalDiagM1,
                    montyInverse
                );
                state->index = 0;
            }
        }
    }

    __device__ void absorbRow(
        Matrix<F>* in,
        int row_idx,
        HasherState<Params>* state,
        pF internalRoundConstants,
        pF externalRoundConstants,
        pF matInternalDiagM1,
        F montyInverse
    ) {
        if (in->row_major) {
            F* row = &in->values[in->width * row_idx];
            absorb(
                row,
                in->width,
                state,
                internalRoundConstants,
                externalRoundConstants,
                matInternalDiagM1,
                montyInverse
            );
        } else {
            for (int j = 0; j < in->width; j++) {
                absorb(
                    &in->values[j * in->height + row_idx],
                    1,
                    state,
                    internalRoundConstants,
                    externalRoundConstants,
                    matInternalDiagM1,
                    montyInverse
                );
            }
        }
    }

    __device__ void finalize(
        HasherState<Params>* state,
        F out[Params::DIGEST_WIDTH],
        pF internalRoundConstants,
        pF externalRoundConstants,
        pF matInternalDiagM1,
        F montyInverse
    ) {
        if (state->index != 0) {
            permute(
                state->data,
                state->data,
                internalRoundConstants,
                externalRoundConstants,
                matInternalDiagM1,
                montyInverse
            );
        }
        for (int i = 0; i < Params::DIGEST_WIDTH; i++) {
            out[i] = state->data[i];
        }
    }
};

template<typename Params>
class DynamicHasher: public Hasher<Params> {
    using F = typename Params::F;
    using pF = typename Params::pF;

  private:
    pF internalRoundConstants;
    pF externalRoundConstants;
    pF matInternalDiagM1;
    F montyInverse;

  public:
    DynamicHasher() {
        cudaMalloc(&internalRoundConstants, Params::ROUNDS_P * sizeof(F));
        cudaMalloc(
            &externalRoundConstants,
            Params::ROUNDS_F * Params::WIDTH * sizeof(F)
        );
        cudaMalloc(&matInternalDiagM1, Params::WIDTH * sizeof(F));
    }

    ~DynamicHasher() {
        cudaFree(internalRoundConstants);
        cudaFree(externalRoundConstants);
        cudaFree(matInternalDiagM1);
    }

    void setParams(
        F (*internalRC)[Params::ROUNDS_P],
        F (*externalRC)[Params::ROUNDS_F * Params::WIDTH],
        F (*internalDiagM1)[Params::WIDTH]
        // F inverse
    ) {
        cudaMemcpy(
            internalRoundConstants,
            internalRC,
            Params::ROUNDS_P * sizeof(F),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            externalRoundConstants,
            externalRC,
            Params::ROUNDS_F * Params::WIDTH * sizeof(F),
            cudaMemcpyHostToDevice
        );
        cudaMemcpy(
            matInternalDiagM1,
            internalDiagM1,
            Params::WIDTH * sizeof(F),
            cudaMemcpyHostToDevice
        );
        // montyInverse = inverse;
    }

    __device__ void permute(F in[Params::WIDTH], F out[Params::WIDTH]) {
        Hasher<Params>::permute(
            in,
            out,
            internalRoundConstants,
            externalRoundConstants,
            matInternalDiagM1,
            montyInverse
        );
    }

    __device__ void compress(
        F left[Params::DIGEST_WIDTH],
        F right[Params::DIGEST_WIDTH],
        F out[Params::DIGEST_WIDTH]
    ) {
        Hasher<Params>::compress(
            left,
            right,
            out,
            internalRoundConstants,
            externalRoundConstants,
            matInternalDiagM1,
            montyInverse
        );
    }

    __device__ void hash(F* in, size_t nIn, F out[Params::DIGEST_WIDTH]) {
        Hasher<Params>::hash(
            in,
            nIn,
            out,
            internalRoundConstants,
            externalRoundConstants,
            matInternalDiagM1,
            montyInverse
        );
    }

    __device__ void absorb(F* in, size_t nIn, HasherState<Params>* state) {
        Hasher<Params>::absorb(
            in,
            nIn,
            state,
            internalRoundConstants,
            externalRoundConstants,
            matInternalDiagM1,
            montyInverse
        );
    }

    __device__ void
    absorbRow(Matrix<F>* in, int row_idx, HasherState<Params>* state) {
        Hasher<Params>::absorbRow(
            in,
            row_idx,
            state,
            internalRoundConstants,
            externalRoundConstants,
            matInternalDiagM1,
            montyInverse
        );
    }

    __device__ void
    finalize(HasherState<Params>* state, F out[Params::DIGEST_WIDTH]) {
        Hasher<Params>::finalize(
            state,
            out,
            internalRoundConstants,
            externalRoundConstants,
            matInternalDiagM1,
            montyInverse
        );
    }
};

template<typename Params>
class StaticHasher: public Hasher<Params> {
    using F = typename Params::F;

  public:
    __device__ void permute(F in[Params::WIDTH], F out[Params::WIDTH]) {
        Hasher<Params>::permute(
            in,
            out,
            Params::INTERNAL_ROUND_CONSTANTS,
            Params::EXTERNAL_ROUND_CONSTANTS,
            Params::MAT_INTERNAL_DIAG_M1,
            Params::MONTY_INVERSE
        );
    }

    __device__ void compress(
        F left[Params::DIGEST_WIDTH],
        F right[Params::DIGEST_WIDTH],
        F out[Params::DIGEST_WIDTH]
    ) {
        Hasher<Params>::compress(
            left,
            right,
            out,
            Params::INTERNAL_ROUND_CONSTANTS,
            Params::EXTERNAL_ROUND_CONSTANTS,
            Params::MAT_INTERNAL_DIAG_M1,
            Params::MONTY_INVERSE
        );
    }

    __device__ void hash(F* in, size_t nIn, F out[Params::DIGEST_WIDTH]) {
        Hasher<Params>::hash(
            in,
            nIn,
            out,
            Params::INTERNAL_ROUND_CONSTANTS,
            Params::EXTERNAL_ROUND_CONSTANTS,
            Params::MAT_INTERNAL_DIAG_M1,
            Params::MONTY_INVERSE
        );
    }

    __device__ void absorb(F* in, size_t nIn, HasherState<Params>* state) {
        Hasher<Params>::absorb(
            in,
            nIn,
            state,
            Params::INTERNAL_ROUND_CONSTANTS,
            Params::EXTERNAL_ROUND_CONSTANTS,
            Params::MAT_INTERNAL_DIAG_M1,
            Params::MONTY_INVERSE
        );
    }

    __device__ void
    absorbRow(Matrix<F>* in, int row_idx, HasherState<Params>* state) {
        Hasher<Params>::absorbRow(
            in,
            row_idx,
            state,
            Params::INTERNAL_ROUND_CONSTANTS,
            Params::EXTERNAL_ROUND_CONSTANTS,
            Params::MAT_INTERNAL_DIAG_M1,
            Params::MONTY_INVERSE
        );
    }

    __device__ void
    finalize(HasherState<Params>* state, F out[Params::DIGEST_WIDTH]) {
        Hasher<Params>::finalize(
            state,
            out,
            Params::INTERNAL_ROUND_CONSTANTS,
            Params::EXTERNAL_ROUND_CONSTANTS,
            Params::MAT_INTERNAL_DIAG_M1,
            Params::MONTY_INVERSE
        );
    }
};

}  // namespace poseidon2
