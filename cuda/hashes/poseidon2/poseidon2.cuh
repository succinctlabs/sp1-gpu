#pragma once

#include "../../matrix/matrix.cuh"
#include "../../utils/vector.cuh"

namespace poseidon2 {

template<typename Params>
struct HasherState {
    using F_t = typename Params::F_t;

    F_t data[Params::WIDTH];
    size_t index;

    __device__ HasherState() : index(0) {
        for (int i = 0; i < Params::WIDTH; ++i) {
            data[i].zero();
        }
    }
};

template<typename Params>
class Hasher {
    using F_t = typename Params::F_t;
    using pF_t = typename Params::pF_t;

  private:
    __device__ static void addExtRc(F_t state[Params::WIDTH], pF_t rc) {
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i] += rc[i];
        }
    }

    __device__ static void sbox(F_t state[Params::WIDTH]) {
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i] ^= Params::D;
        }
    }

  public:
    // TODO: are we sacrificing infornation about the length of the params?
    // TODO: poseidon2 params should be passed around more cleanly
    __device__ static void permute(
        F_t in[Params::WIDTH],
        F_t out[Params::WIDTH],
        pF_t internalRoundConstants,
        pF_t externalRoundConstants,
        pF_t matInternalDiagM1,
        F_t montyInverse
    ) {
        F_t state[Params::WIDTH];
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i] = in[i];
        }

        Params::externalLinearLayer(state);

        int rounds_f_half = Params::ROUNDS_F / 2;
        for (int i = 0; i < rounds_f_half; i++) {
            addExtRc(state, externalRoundConstants + i * Params::WIDTH);
            sbox(state);
            Params::externalLinearLayer(state);
        }

        for (int i = 0; i < Params::ROUNDS_P; i++) {
            state[0] += internalRoundConstants[i];
            state[0] ^= Params::D;
            Params::internalLinearLayer(state, matInternalDiagM1, montyInverse);
        }

        for (int i = rounds_f_half; i < Params::ROUNDS_F; i++) {
            addExtRc(state, externalRoundConstants + i * Params::WIDTH);
            sbox(state);
            Params::externalLinearLayer(state);
        }

        for (int i = 0; i < Params::WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ static void compress(
        F_t left[Params::DIGEST_WIDTH],
        F_t right[Params::DIGEST_WIDTH],
        F_t out[Params::DIGEST_WIDTH],
        pF_t internalRoundConstants,
        pF_t externalRoundConstants,
        pF_t matInternalDiagM1,
        F_t montyInverse
    ) {
        F_t state[Params::WIDTH];
        for (int i = 0; i < Params::DIGEST_WIDTH; i++) {
            state[i] = left[i];
            state[i + Params::DIGEST_WIDTH] = right[i];
        }
        for (int i = 2 * Params::DIGEST_WIDTH; i < Params::WIDTH; i++) {
            state[i].zero();
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

    __device__ static void hash(
        F_t* in,
        size_t nIn,
        F_t out[Params::DIGEST_WIDTH],
        pF_t internalRoundConstants,
        pF_t externalRoundConstants,
        pF_t matInternalDiagM1,
        F_t montyInverse
    ) {
        F_t state[Params::WIDTH];
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

    __device__ static void absorb(
        F_t* in,
        size_t nIn,
        HasherState<Params>* state,
        pF_t internalRoundConstants,
        pF_t externalRoundConstants,
        pF_t matInternalDiagM1,
        F_t montyInverse
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

    __device__ static void absorbRow(
        Matrix<F_t>* in,
        int row_idx,
        HasherState<Params>* state,
        pF_t internalRoundConstants,
        pF_t externalRoundConstants,
        pF_t matInternalDiagM1,
        F_t montyInverse
    ) {
        if (in->row_major) {
            F_t* row = &in->values[in->width * row_idx];
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

    __device__ static void finalize(
        HasherState<Params>* state,
        F_t out[Params::DIGEST_WIDTH],
        pF_t internalRoundConstants,
        pF_t externalRoundConstants,
        pF_t matInternalDiagM1,
        F_t montyInverse
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
    using F_t = typename Params::F_t;
    using pF_t = typename Params::pF_t;

  private:
    pF_t internalRoundConstants;
    pF_t externalRoundConstants;
    pF_t matInternalDiagM1;
    F_t montyInverse;

  public:
    DynamicHasher() {
        cudaMalloc(&internalRoundConstants, Params::ROUNDS_P * sizeof(F_t));
        cudaMalloc(
            &externalRoundConstants,
            Params::ROUNDS_F * Params::WIDTH * sizeof(F_t)
        );
        cudaMalloc(&matInternalDiagM1, Params::WIDTH * sizeof(F_t));
    }

    ~DynamicHasher() {
        cudaFree(internalRoundConstants);
        cudaFree(externalRoundConstants);
        cudaFree(matInternalDiagM1);
    }

    void setInternalRoundConstants(F_t (*internalRC)[Params::ROUNDS_P]) {
        cudaMemcpy(
            internalRoundConstants,
            internalRC,
            Params::ROUNDS_P * sizeof(F_t),
            cudaMemcpyHostToDevice
        );
    }

    void setExternalRoundConstants(F_t (*externalRC
    )[Params::ROUNDS_F * Params::WIDTH]) {
        cudaMemcpy(
            externalRoundConstants,
            externalRC,
            Params::ROUNDS_F * Params::WIDTH * sizeof(F_t),
            cudaMemcpyHostToDevice
        );
    }

    void setMatInternalDiagM1(F_t (*internalDiagM1)[Params::WIDTH]) {
        cudaMemcpy(
            matInternalDiagM1,
            internalDiagM1,
            Params::WIDTH * sizeof(F_t),
            cudaMemcpyHostToDevice
        );
    }

    void setMontyInverse(F_t montyInverse) {
        this->montyInverse = montyInverse;
    }

    __device__ void permute(F_t in[Params::WIDTH], F_t out[Params::WIDTH]) {
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
        F_t left[Params::DIGEST_WIDTH],
        F_t right[Params::DIGEST_WIDTH],
        F_t out[Params::DIGEST_WIDTH]
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

    __device__ void hash(F_t* in, size_t nIn, F_t out[Params::DIGEST_WIDTH]) {
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

    __device__ void absorb(F_t* in, size_t nIn, HasherState<Params>* state) {
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
    absorbRow(Matrix<F_t>* in, int row_idx, HasherState<Params>* state) {
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
    finalize(HasherState<Params>* state, F_t out[Params::DIGEST_WIDTH]) {
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
    using F_t = typename Params::F_t;

  public:
    __device__ static void
    permute(F_t in[Params::WIDTH], F_t out[Params::WIDTH]) {
        Hasher<Params>::permute(
            in,
            out,
            Params::INTERNAL_ROUND_CONSTANTS,
            Params::EXTERNAL_ROUND_CONSTANTS,
            Params::MAT_INTERNAL_DIAG_M1,
            Params::MONTY_INVERSE
        );
    }

    __device__ static void compress(
        F_t left[Params::DIGEST_WIDTH],
        F_t right[Params::DIGEST_WIDTH],
        F_t out[Params::DIGEST_WIDTH]
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

    __device__ static void
    hash(F_t* in, size_t nIn, F_t out[Params::DIGEST_WIDTH]) {
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

    __device__ static void
    absorb(F_t* in, size_t nIn, HasherState<Params>* state) {
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

    __device__ static void
    absorbRow(Matrix<F_t>* in, int row_idx, HasherState<Params>* state) {
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

    __device__ static void
    finalize(HasherState<Params>* state, F_t out[Params::DIGEST_WIDTH]) {
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
