#include <cstdint>

#include "../../utils/constants.hpp"

__uint128_t to_bigint(uint32_t small_ints[INTS_PER_VALUE]);

void to_bigint_arr(uint32_t *small_ints, __uint128_t *big_ints, size_t num_big_ints);