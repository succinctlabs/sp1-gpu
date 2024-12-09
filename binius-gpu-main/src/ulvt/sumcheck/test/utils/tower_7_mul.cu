#include "tower_7_mul.cuh"
#include "unbitsliced_mul.cuh"

__host__ __device__ __uint128_t tower_height_7_mul(const __uint128_t a, const __uint128_t b) {
	uint64_t a1 = (uint64_t)(a >> 64);
	uint64_t a0 = (uint64_t)(a & ((__uint128_t)0xffffffffffffffff));

	uint64_t b1 = (uint64_t)(b >> 64);
	uint64_t b0 = (uint64_t)(b & ((__uint128_t)0xffffffffffffffff));

	uint64_t a0b0 = FanPaarTowerField<6>::multiply(a0, b0);
	uint64_t a0b1 = FanPaarTowerField<6>::multiply(a0, b1);
	uint64_t a1b0 = FanPaarTowerField<6>::multiply(a1, b0);
	uint64_t a1b1 = FanPaarTowerField<6>::multiply(a1, b1);

	uint64_t result_bottom_half = a0b0 ^ a1b1;
	uint64_t result_top_half = a0b1 ^ a1b0 ^ FanPaarTowerField<6>::multiply_alpha(a1b1);

	return ((__uint128_t)result_top_half) << 64 | ((__uint128_t)result_bottom_half);
}

__uint128_t inverse_at_interpolation_point(const __uint128_t x) {
	return ((__uint128_t)FanPaarTowerField<2>::inverse((uint64_t)x));
}
