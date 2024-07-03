#pragma once

#include "../../matrix/matrix.cuh"
#include "../../utils/vector.cuh"
#include "poseidon2_bb31_16.cuh"
#include "poseidon2_bn254_3.cuh"

namespace poseidon2 {

template<typename Params>
struct RoundConstants {
    using F_t = typename Params::F_t;
    using pF_t = typename Params::pF_t;

    pF_t* internalRoundConstants;
    pF_t* externalRoundConstants;
    pF_t* matInternalDiagM1;
    pF_t montyInverse;
};

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

template<typename Params, typename P_t, int R>
struct MultiFieldHasherState: public HasherState<Params> {
    P_t overhang[R];
    size_t overhangSize;
};

template<typename Params, typename HasherState_t>
class Hasher {
    using F_t = typename Params::F_t;
    using pF_t = typename Params::pF_t;
    using RoundConstants_t = RoundConstants<Params>;

  private:
    __device__ static void
    addExtRc(F_t state[Params::WIDTH], pF_t rc[Params::WIDTH]) {
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
    __device__ static void permute(
        F_t in[Params::WIDTH],
        F_t out[Params::WIDTH],
        RoundConstants_t roundConstants
    ) {
        F_t state[Params::WIDTH];
        for (int i = 0; i < Params::WIDTH; i++) {
            state[i] = in[i];
        }

        Params::externalLinearLayer(state);

        int rounds_f_half = Params::ROUNDS_F / 2;
        for (int i = 0; i < rounds_f_half; i++) {
            addExtRc(
                state,
                roundConstants.externalRoundConstants + i * Params::WIDTH
            );
            sbox(state);
            Params::externalLinearLayer(state);
        }

        for (int i = 0; i < Params::ROUNDS_P; i++) {
            state[0] += roundConstants.internalRoundConstants[i];
            state[0] ^= Params::D;
            Params::internalLinearLayer(
                state,
                roundConstants.matInternalDiagM1,
                roundConstants.montyInverse
            );
        }

        for (int i = rounds_f_half; i < Params::ROUNDS_F; i++) {
            addExtRc(
                state,
                roundConstants.externalRoundConstants + i * Params::WIDTH
            );
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
        RoundConstants_t roundConstants
    ) {
        F_t state[Params::WIDTH];
        for (int i = 0; i < Params::DIGEST_WIDTH; i++) {
            state[i] = left[i];
            state[i + Params::DIGEST_WIDTH] = right[i];
        }
        for (int i = 2 * Params::DIGEST_WIDTH; i < Params::WIDTH; i++) {
            state[i].zero();
        }
        permute(state, state, roundConstants);
        for (int i = 0; i < Params::DIGEST_WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ static void hash(
        F_t* in,
        size_t nIn,
        F_t out[Params::DIGEST_WIDTH],
        RoundConstants_t roundConstants
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
            permute(state, state, roundConstants);
        }

        for (int i = 0; i < Params::DIGEST_WIDTH; i++) {
            out[i] = state[i];
        }
    }

    __device__ static void absorb(
        F_t* in,
        size_t nIn,
        HasherState_t* state,
        RoundConstants_t roundConstants
    ) {
        for (int i = 0; i < nIn; i++) {
            state->data[state->index] = in[i];
            state->index++;
            if (state->index == Params::RATE) {
                permute(state->data, state->data, roundConstants);
                state->index = 0;
            }
        }
    }

    // __device__ static void absorbRow(
    //     Matrix<F_t>* in,
    //     int row_idx,
    //     HasherState_t* state,
    //     RoundConstants_t roundConstants
    // ) {
    //     if (in->row_major) {
    //         F_t* row = &in->values[in->width * row_idx];
    //         absorb(row, in->width, state, roundConstants);
    //     } else {
    //         for (int j = 0; j < in->width; j++) {
    //             absorb(
    //                 &in->values[j * in->height + row_idx],
    //                 1,
    //                 state,
    //                 roundConstants
    //             );
    //         }
    //     }
    // }

    __device__ static void finalize(
        HasherState_t* state,
        F_t out[Params::DIGEST_WIDTH],
        RoundConstants_t roundConstants
    ) {
        if (state->index != 0) {
            permute(state->data, state->data, roundConstants);
        }
        for (int i = 0; i < Params::DIGEST_WIDTH; i++) {
            out[i] = state->data[i];
        }
    }
};

template<typename Params, typename HasherState_t>
class DynamicHasher: public Hasher<Params, HasherState_t> {
    using F_t = typename Params::F_t;
    using pF_t = typename Params::pF_t;
    using Hasher_t = Hasher<Params, HasherState_t>;

  public:
    RoundConstants<Params> roundConstants;

    void setInternalRoundConstants(pF_t* internalRoundConstants) {
        roundConstants.internalRoundConstants = internalRoundConstants;
    }

    void setExternalRoundConstants(pF_t* externalRoundConstants) {
        roundConstants.externalRoundConstants = externalRoundConstants;
    }

    void setMatInternalDiagM1(pF_t* matInternalDiagM1) {
        roundConstants.matInternalDiagM1 = matInternalDiagM1;
    }

    void setMontyInverse(pF_t montyInverse) {
        roundConstants.montyInverse = montyInverse;
    }

    __device__ void permute(F_t in[Params::WIDTH], F_t out[Params::WIDTH]) {
        Hasher_t::permute(in, out, roundConstants);
    }

    __device__ void compress(
        F_t left[Params::DIGEST_WIDTH],
        F_t right[Params::DIGEST_WIDTH],
        F_t out[Params::DIGEST_WIDTH]
    ) {
        Hasher_t::compress(left, right, out, roundConstants);
    }

    __device__ void hash(F_t* in, size_t nIn, F_t out[Params::DIGEST_WIDTH]) {
        Hasher_t::hash(in, nIn, out, roundConstants);
    }

    __device__ void absorb(F_t* in, size_t nIn, HasherState_t* state) {
        Hasher_t::absorb(in, nIn, state, roundConstants);
    }

    __device__ void
    finalize(HasherState_t* state, F_t out[Params::DIGEST_WIDTH]) {
        // TODO: Make this cleaner
        if (state->overhangSize != 0) {
            // F_t value = poseidon2_bn254_3::reduceBabyBear(
            //     state->overhang,
            //     state->overhangSize,
            //     1
            // );
            F_t value;
            value.zero();
            absorb(&value, 1, state);
        }
        Hasher_t::finalize(state, out, roundConstants);
    }
};

template<typename Params, typename HasherState_t>
class StaticHasher: public Hasher<Params, HasherState_t> {
    using F_t = typename Params::F_t;
    using Hasher_t = Hasher<Params, HasherState_t>;
    using RoundConstants_t = RoundConstants<Params>;

  public:
    static constexpr const RoundConstants_t roundConstants = {
        Params::INTERNAL_ROUND_CONSTANTS,
        Params::EXTERNAL_ROUND_CONSTANTS,
        Params::MAT_INTERNAL_DIAG_M1,
        Params::MONTY_INVERSE
    };

    __device__ static void
    permute(F_t in[Params::WIDTH], F_t out[Params::WIDTH]) {
        Hasher_t::permute(in, out, roundConstants);
    }

    __device__ static void compress(
        F_t left[Params::DIGEST_WIDTH],
        F_t right[Params::DIGEST_WIDTH],
        F_t out[Params::DIGEST_WIDTH]
    ) {
        Hasher_t::compress(left, right, out, roundConstants);
    }

    __device__ static void
    hash(F_t* in, size_t nIn, F_t out[Params::DIGEST_WIDTH]) {
        Hasher_t::hash(in, nIn, out, roundConstants);
    }

    __device__ static void absorb(F_t* in, size_t nIn, HasherState_t* state) {
        Hasher_t::absorb(in, nIn, state, roundConstants);
    }

    __device__ static void
    finalize(HasherState_t* state, F_t out[Params::DIGEST_WIDTH]) {
        Hasher_t::finalize(state, out, roundConstants);
    }
};

template<typename Params>
class Bn254Hasher:
    public DynamicHasher<Params, MultiFieldHasherState<Params, bb31_t, 8>> {
  public:
    __device__ void absorbRow(
        Matrix<bb31_t>* in,
        int row_idx,
        MultiFieldHasherState<Params, bb31_t, 8>* state
    ) {
        poseidon2_bn254_3::absorbRow<
            Bn254Hasher<Params>,
            MultiFieldHasherState<Params, bb31_t, 8>>(
            *this,
            in,
            row_idx,
            state
        );
    }
};

template<typename Params>
class BabyBearHasher: public StaticHasher<Params, HasherState<Params>> {
  public:
    __device__ void
    absorbRow(Matrix<bb31_t>* in, int row_idx, HasherState<Params>* state) {
        poseidon2_bb31_16::absorbRow<
            BabyBearHasher<Params>,
            HasherState<Params>>(*this, in, row_idx, state);
    }
};

}  // namespace poseidon2
