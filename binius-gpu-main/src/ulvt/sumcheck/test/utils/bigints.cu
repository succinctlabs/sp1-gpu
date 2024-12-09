#include <cstdint>

#include "../../utils/constants.hpp"
#include "bigints.cuh"

__uint128_t to_bigint(uint32_t small_ints[INTS_PER_VALUE]) {
	__uint128_t result = 0;
	for (int i = INTS_PER_VALUE - 1; i >= 0; --i) {
		result <<= 32;
		result |= small_ints[i];
	}
	return result;
}

void to_bigint_arr(uint32_t *small_ints, __uint128_t *big_ints, size_t num_big_ints) {
	for (size_t idx_in_bigint_arr = 0; idx_in_bigint_arr < num_big_ints; ++idx_in_bigint_arr) {
		big_ints[idx_in_bigint_arr] = to_bigint(small_ints + INTS_PER_VALUE * idx_in_bigint_arr);
	}
}