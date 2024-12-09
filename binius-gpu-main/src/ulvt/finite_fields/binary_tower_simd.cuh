#pragma once

#include <array>
#include <cstdint>
#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/utility>

/** Declarations */

template <size_t HEIGHT>
static __host__ __device__ uint32_t mul_binary_tower_32b_simd(uint32_t a, uint32_t b);

template <size_t HEIGHT>
__host__ __device__ cuda::std::pair<uint32_t, uint32_t> interleave_32b(uint32_t a, uint32_t b);

template <size_t HEIGHT>
__host__ __device__ uint32_t xor_adjacent_32b(uint32_t a);

template <size_t LOG_BITS>
struct PackedBinaryField {
	cuda::std::array<uint32_t, 1 << (LOG_BITS - 5)> words;

	__host__ __device__ PackedBinaryField() { static_assert(LOG_BITS >= 5, "Minimum LOG_BITS supported is 5"); }

	__host__ __device__ bool operator==(const PackedBinaryField<LOG_BITS>& rhs) const {
		if (words.size() != rhs.words.size()) return false;
		for (int i = 0; i < words.size(); i++) {
			if (words[i] != rhs.words[i]) return false;
		}
		return true;
	}
};

/** Implementations */

const std::array<uint32_t, 5> HOST_MASKS{
	0x55555555,
	0x33333333,
	0x0f0f0f0f,
	0x00ff00ff,
	0x0000ffff,
};

__constant__ const uint32_t DEVICE_MASKS[5] = {
	0x55555555,
	0x33333333,
	0x0f0f0f0f,
	0x00ff00ff,
	0x0000ffff,
};

const std::array<uint32_t, 5> HOST_ALPHAS{
	0x55555555,
	0x22222222,
	0x04040404,
	0x00100010,
	0x00000100,
};

__constant__ const uint32_t DEVICE_ALPHAS[5] = {
	0x55555555,
	0x22222222,
	0x04040404,
	0x00100010,
	0x00000100,
};

#ifdef __CUDA_ARCH__
#define MASKS DEVICE_MASKS
#define ALPHAS DEVICE_ALPHAS
#else
#define MASKS HOST_MASKS
#define ALPHAS HOST_ALPHAS
#endif

template <>
__host__ __device__ uint32_t mul_binary_tower_32b_simd<0>(uint32_t a, uint32_t b) {
	return a & b;
}

template <size_t HEIGHT>
static __host__ __device__ uint32_t mul_binary_tower_32b_simd(uint32_t a, uint32_t b) {
	// a and b can be interpreted as packed subfield elements:
	// a = <a_lo_0, a_hi_0, a_lo_1, a_hi_1, ...>
	// b = <b_lo_0, b_hi_0, b_lo_1, b_hi_1, ...>

	// ab is the product of a * b as packed subfield elements
	// ab = <a_lo_0 * b_lo_0, a_hi_0 * b_hi_0, a_lo_1 * b_lo_1, a_hi_1 * b_hi_1, ...>
	auto z0_even_z2_odd = mul_binary_tower_32b_simd<HEIGHT - 1>(a, b);

	// lo = <a_lo_0, b_lo_0, a_lo_1, b_lo_1, ...>
	// hi = <a_hi_0, b_hi_0, a_hi_1, b_hi_1, ...>
	uint32_t lo, hi;
	cuda::std::tie(lo, hi) = interleave_32b<HEIGHT - 1>(a, b);

	// <a_lo_0 + a_hi_0, b_lo_0 + b_hi_0, a_lo_1 + a_hi_1, b_lo_1 + b_hi_1, ...>
	auto lo_plus_hi_a_even_b_odd = lo ^ hi;

	auto even_mask = MASKS[HEIGHT - 1];
	auto alphas = ALPHAS[HEIGHT - 1];
	auto block_len = 1 << (HEIGHT - 1);
	auto odd_mask = even_mask << block_len;

	// <α, z2_0, α, z2_1, ...>
	auto alpha_even_z2_odd = alphas ^ (z0_even_z2_odd & odd_mask);

	// a_lo_plus_hi_even_z2_odd    = <a_lo_0 + a_hi_0, z2_0, a_lo_1 + a_hi_1, z2_1, ...>
	// b_lo_plus_hi_even_alpha_odd = <b_lo_0 + b_hi_0,    α, a_lo_1 + a_hi_1,   αz, ...>
	uint32_t a_lo_plus_hi_even_alpha_odd, b_lo_plus_hi_even_z2_odd;
	cuda::std::tie(a_lo_plus_hi_even_alpha_odd, b_lo_plus_hi_even_z2_odd) =
		interleave_32b<HEIGHT - 1>(lo_plus_hi_a_even_b_odd, alpha_even_z2_odd);

	// <z1_0 + z0_0 + z2_0, z2a_0, z1_1 + z0_1 + z2_1, z2a_1, ...>
	auto z1_plus_z0_plus_z2_even_z2a_odd =
		mul_binary_tower_32b_simd<HEIGHT - 1>(a_lo_plus_hi_even_alpha_odd, b_lo_plus_hi_even_z2_odd);

	// <0, z1_0 + z2a_0 + z0_0 + z2_0, 0, z1_1 + z2a_1 + z0_1 + z2_1, ...>
	auto zero_even_z1_plus_z2a_plus_z0_plus_z2_odd =
		(z1_plus_z0_plus_z2_even_z2a_odd ^ (z1_plus_z0_plus_z2_even_z2a_odd << block_len)) & odd_mask;

	// <z0_0 + z2_0, z0_0 + z2_0, z0_1 + z2_1, z0_1 + z2_1, ...>
	auto z0_plus_z2_dup = xor_adjacent_32b<HEIGHT - 1>(z0_even_z2_odd);

	// <z0_0 + z2_0, z1_0 + z2a_0, z0_1 + z2_1, z1_1 + z2a_1, ...>
	return z0_plus_z2_dup ^ zero_even_z1_plus_z2a_plus_z0_plus_z2_odd;
}

template <size_t HEIGHT>
__host__ __device__ cuda::std::pair<uint32_t, uint32_t> interleave_32b(uint32_t a, uint32_t b) {
	static_assert(HEIGHT < 5, "interleave_32b requires tower height < 5");

	auto mask = MASKS[HEIGHT];
	auto block_len = 1 << HEIGHT;
	auto t = ((a >> block_len) ^ b) & mask;
	auto c = a ^ (t << block_len);
	auto d = b ^ t;
	return cuda::std::make_pair(c, d);
}

template <size_t HEIGHT>
__host__ __device__ uint32_t xor_adjacent_32b(uint32_t a) {
	static_assert(HEIGHT < 5, "xor_adjacent_32b requires tower height < 5");

	auto mask = MASKS[HEIGHT];
	auto block_len = 1 << HEIGHT;
	auto t = ((a >> block_len) ^ a) & mask;
	return t ^ (t << block_len);
}

template <size_t HEIGHT, size_t LOG_BITS>
__host__ __device__ cuda::std::pair<PackedBinaryField<LOG_BITS>, PackedBinaryField<LOG_BITS>> interleave(
	PackedBinaryField<LOG_BITS> a, PackedBinaryField<LOG_BITS> b
) {
	static_assert(HEIGHT < LOG_BITS, "interleave requires tower height < LOG_BITS");

	if (HEIGHT < 5) {
		PackedBinaryField<LOG_BITS> c, d;
		for (size_t i = 0; i < 1 << (LOG_BITS - 5); i++) {
			auto c_d_i = interleave_32b<HEIGHT>(a.words[i], b.words[i]);
			c.words[i] = c_d_i.first;
			d.words[i] = c_d_i.second;
		}
		return cuda::std::make_pair(c, d);
	} else {
		PackedBinaryField<LOG_BITS> c, d;
		auto block_len = 1 << (HEIGHT - 5);
		for (size_t i = 0; i < 1 << (LOG_BITS - HEIGHT); i++) {
#pragma unroll
			for (size_t j = 0; j < block_len; j++) {
				c.words[i] = a.words[i];
			}
#pragma unroll
			for (size_t j = 0; j < block_len; j++) {
				c.words[i + block_len] = b.words[i];
			}
#pragma unroll
			for (size_t j = 0; j < block_len; j++) {
				d.words[i] = b.words[i + block_len];
			}
#pragma unroll
			for (size_t j = 0; j < block_len; j++) {
				d.words[i + block_len] = a.words[i + block_len];
			}
		}
		return cuda::std::make_pair(c, d);
	}
}
