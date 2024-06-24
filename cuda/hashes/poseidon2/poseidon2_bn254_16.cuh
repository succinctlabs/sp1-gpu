#pragma once

#include "../../fields/bn254.cuh"

namespace poseidon2_bn254_16 {

namespace constants {

    constexpr int DIGEST_WIDTH = 1;  // ?
    constexpr int RATE = 1;  // ?
    constexpr int WIDTH = 3;
    constexpr int ROUNDS_P = 56;
    constexpr int ROUNDS_F = 8;
    constexpr int D = 5;

    // Missing constants [!]

    __constant__ bn254_t INTERNAL_ROUND_CONSTANTS[ROUNDS_P] = {};

    __constant__ bn254_t EXTERNAL_ROUND_CONSTANTS[ROUNDS_F][WIDTH] = {};

    __constant__ bn254_t
        MAT_INTERNAL_DIAG_M1[WIDTH] = {bn254_t(1), bn254_t(1), bn254_t(2)};

    __constant__ bn254_t MONTY_INVERSE;
}  // namespace constants

class BarretoNaehrig16 {
  public:
    using F = bn254_t;

    static constexpr int DIGEST_WIDTH = constants::DIGEST_WIDTH;
    static constexpr int RATE = constants::RATE;
    static constexpr int WIDTH = constants::WIDTH;
    static constexpr int ROUNDS_F = constants::ROUNDS_F;
    static constexpr int ROUNDS_P = constants::ROUNDS_P;

    __device__ static constexpr const int getD() {
        return constants::D;
    }

    __device__ static constexpr const F* getInternalRoundConstants() {
        return constants::INTERNAL_ROUND_CONSTANTS;
    }

    __device__ static constexpr const F (*getExternalRoundConstants())[WIDTH] {
        return constants::EXTERNAL_ROUND_CONSTANTS;
    }

    __device__ static constexpr const F* getMatInternalDiagM1() {
        return constants::MAT_INTERNAL_DIAG_M1;
    }

    __device__ static constexpr const F& getMontyInverse() {
        return constants::MONTY_INVERSE;
    }
};

}  // namespace poseidon2_bn254_16
