#pragma once

#include "../../fields/bn254_t.cuh"

namespace poseidon2_bn254_16 {

namespace constants {

    constexpr const int DIGEST_WIDTH = 1;  // ?
    constexpr const int RATE = 1;  // ?
    constexpr const int WIDTH = 3;
    constexpr const int ROUNDS_P = 3;  // 56;
    constexpr const int ROUNDS_F = 3;  // 8;
    constexpr const int D = 5;

#define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64 >> 32)

    __constant__ constexpr const bn254_t INTERNAL_ROUND_CONSTANTS[ROUNDS_P] = {
        {TO_CUDA_T(0x0000000000000000)},
        {TO_CUDA_T(0x0000000000000000)},
        {TO_CUDA_T(0x0000000000000000)}
    };

    __constant__ constexpr const bn254_t
        EXTERNAL_ROUND_CONSTANTS[ROUNDS_F][WIDTH] = {
            {{TO_CUDA_T(0x0000000000000000)},
             {TO_CUDA_T(0x0000000000000000)},
             {TO_CUDA_T(0x0000000000000000)}},
            {{TO_CUDA_T(0x0000000000000000)},
             {TO_CUDA_T(0x0000000000000000)},
             {TO_CUDA_T(0x0000000000000000)}},
            {{TO_CUDA_T(0x0000000000000000)},
             {TO_CUDA_T(0x0000000000000000)},
             {TO_CUDA_T(0x0000000000000000)}}
    };

    __constant__ constexpr const bn254_t MAT_INTERNAL_DIAG_M1[WIDTH] = {
        {TO_CUDA_T(0x0000000000000000)},
        {TO_CUDA_T(0x0000000000000000)},
        {TO_CUDA_T(0x0000000000000000)}
    };

    __constant__ constexpr const bn254_t MONTY_INVERSE;

}  // namespace constants

class BarretoNaehrig16 {
  public:
    using F = bn254_t;

    static constexpr const int DIGEST_WIDTH = constants::DIGEST_WIDTH;
    static constexpr const int RATE = constants::RATE;
    static constexpr const int WIDTH = constants::WIDTH;
    static constexpr const int ROUNDS_F = constants::ROUNDS_F;
    static constexpr const int ROUNDS_P = constants::ROUNDS_P;
    static constexpr const int D = constants::D;

    static constexpr const F* INTERNAL_ROUND_CONSTANTS =
        constants::INTERNAL_ROUND_CONSTANTS;
    static constexpr const F (*EXTERNAL_ROUND_CONSTANTS
    )[WIDTH] = constants::EXTERNAL_ROUND_CONSTANTS;
    static constexpr const F* MAT_INTERNAL_DIAG_M1 =
        constants::MAT_INTERNAL_DIAG_M1;
    static constexpr const F& MONTY_INVERSE = constants::MONTY_INVERSE;
};

}  // namespace poseidon2_bn254_16
