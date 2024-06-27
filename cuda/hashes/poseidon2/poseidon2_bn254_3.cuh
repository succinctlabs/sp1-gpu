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

// TODO: rename
class BarretoNaehrig3 {
  public:
    using F_t = bn254_t;  // TODO: rename to F_t
    using pF_t = F_t*;

    static constexpr const int DIGEST_WIDTH = constants::DIGEST_WIDTH;
    static constexpr const int RATE = constants::RATE;
    static constexpr const int WIDTH = constants::WIDTH;
    static constexpr const int ROUNDS_F = constants::ROUNDS_F;
    static constexpr const int ROUNDS_P = constants::ROUNDS_P;
    static constexpr const int D = constants::D;
};

}  // namespace poseidon2_bn254_3
