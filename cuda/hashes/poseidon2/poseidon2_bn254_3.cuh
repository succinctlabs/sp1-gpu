#pragma once

#include "../../fields/bn254_t.cuh"

namespace poseidon2_bn254_3 {

namespace constants {

    constexpr const int DIGEST_WIDTH = 1;  // ?
    constexpr const int RATE = 1;  // ?
    constexpr const int WIDTH = 3;
    constexpr const int ROUNDS_P = 56;
    constexpr const int ROUNDS_F = 8;
    constexpr const int D = 5;

}  // namespace constants

class Bn254 {
  public:
    using F_t = bn254_t;
    using pF_t = F_t*;

    static constexpr const int DIGEST_WIDTH = constants::DIGEST_WIDTH;
    static constexpr const int RATE = constants::RATE;
    static constexpr const int WIDTH = constants::WIDTH;
    static constexpr const int ROUNDS_F = constants::ROUNDS_F;
    static constexpr const int ROUNDS_P = constants::ROUNDS_P;
    static constexpr const int D = constants::D;

    __device__ static void
    internalLinearLayer(F_t state[WIDTH], pF_t matInternalDiagM1, F_t _) {
        F_t s = state[0] + state[1] + state[2];
        for (int i = 0; i < WIDTH; i++) {
            state[i] *= matInternalDiagM1[i];
            state[i] += s;
        }
    }

    __device__ static void externalLinearLayer(F_t state[WIDTH]) {
        F_t sum = state[0] + state[1] + state[2];
        state[0] += sum;
        state[1] += sum;
        state[2] += sum;
    }
};

}  // namespace poseidon2_bn254_3
