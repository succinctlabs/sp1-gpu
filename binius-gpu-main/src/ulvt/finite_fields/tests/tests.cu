#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>

#include "../circuit_generator/unrolled/binary_tower_unrolled.cuh"
#include "../../utils/bitslicing.cuh"
#include "../circuit_generator/utils/utils.hpp"
#include "./profiling/kernels/babybear_repeat.cuh"
#include "./profiling/kernels/bitsliced_repeat.cuh"
#include "ulvt/finite_fields/baby_bear.cuh"
#include "ulvt/finite_fields/binary_tower.cuh"
#include "ulvt/finite_fields/binary_tower_simd.cuh"

TEST_CASE("interleave_32b", "[interleave_32b]") {
	using cuda::std::make_pair;

	uint32_t a, b, c, d;

	a = 0x0000ffff;
	b = 0xffff0000;

	c = 0xaaaa5555;
	d = 0xaaaa5555;
	REQUIRE(interleave_32b<0>(a, b) == make_pair(c, d));
	REQUIRE(interleave_32b<0>(c, d) == make_pair(a, b));

	c = 0xcccc3333;
	d = 0xcccc3333;
	REQUIRE(interleave_32b<1>(a, b) == make_pair(c, d));
	REQUIRE(interleave_32b<1>(c, d) == make_pair(a, b));

	c = 0xf0f00f0f;
	d = 0xf0f00f0f;
	REQUIRE(interleave_32b<2>(a, b) == make_pair(c, d));
	REQUIRE(interleave_32b<2>(c, d) == make_pair(a, b));

	a = 0x03020100;
	b = 0x13121110;

	c = 0x12021000;
	d = 0x13031101;
	REQUIRE(interleave_32b<3>(a, b) == make_pair(c, d));
	REQUIRE(interleave_32b<3>(c, d) == make_pair(a, b));

	c = 0x11100100;
	d = 0x13120302;
	REQUIRE(interleave_32b<4>(a, b) == make_pair(c, d));
	REQUIRE(interleave_32b<4>(c, d) == make_pair(a, b));
}

TEST_CASE("interleave", "[interleave]") {
	using cuda::std::make_pair;

	PackedBinaryField<5> a, b, c, d;
	a.words[0] = 0x0000ffff;
	b.words[0] = 0xffff0000;

	c.words[0] = 0xaaaa5555;
	d.words[0] = 0xaaaa5555;

	auto c_d = interleave<0, 5>(a, b);
	REQUIRE(c_d == make_pair(c, d));
}

TEST_CASE("mul_binary_tower_32b_simd<0>", "[mul]") {
	REQUIRE(mul_binary_tower_32b_simd<0>(0xd82c07cd, 0xd82c07cd) == 0xd82c07cd);
	REQUIRE(mul_binary_tower_32b_simd<0>(0x31a9358b, 0xd82c07cd) == 0x10280589);
	REQUIRE(mul_binary_tower_32b_simd<0>(0x31a9358b, 0x90ec8953) == 0x10a80103);
	REQUIRE(mul_binary_tower_32b_simd<0>(0xcb261cf7, 0xd82c07cd) == 0xc82404c5);
	REQUIRE(mul_binary_tower_32b_simd<0>(0xcb261cf7, 0x3d791a58) == 0x09201850);
}

TEST_CASE("mul_binary_tower_32b_simd<2>", "[mul]") {
	REQUIRE(mul_binary_tower_32b_simd<2>(0xd82c07cd, 0xd82c07cd) == 0xf73e0bef);
	REQUIRE(mul_binary_tower_32b_simd<2>(0x71948b72, 0xd82c07cd) == 0x88e704f6);
	REQUIRE(mul_binary_tower_32b_simd<2>(0x71948b72, 0x8b86a383) == 0xabf1b6a1);
	REQUIRE(mul_binary_tower_32b_simd<2>(0x879f6d99, 0xd82c07cd) == 0x1ae6085c);
	REQUIRE(mul_binary_tower_32b_simd<2>(0x879f6d99, 0x646b38a5) == 0x2547d113);
}

constexpr size_t NUM_OPS = 10000;

TEST_CASE("mul_binary_tower_32b_simd<5>", "[mul]") {
	REQUIRE(mul_binary_tower_32b_simd<5>(0xd82c07cd, 0xd82c07cd) == 0xafab1b8f);
	REQUIRE(mul_binary_tower_32b_simd<5>(0x6b4c9946, 0xd82c07cd) == 0xf35c8d0f);
	REQUIRE(mul_binary_tower_32b_simd<5>(0x6b4c9946, 0x3d47e731) == 0xf849322d);
	REQUIRE(mul_binary_tower_32b_simd<5>(0xbe127079, 0xd82c07cd) == 0xd86f9eba);
	REQUIRE(mul_binary_tower_32b_simd<5>(0xbe127079, 0x2cd911fc) == 0x2b8b8f27);

	uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();

	std::mt19937 generator(seed);
	uint32_t a = generator();

	BENCHMARK("mul_binary_tower_32b_simd cpu") {
		for (size_t i = 0; i < NUM_OPS; i++) {
			a = mul_binary_tower_32b_simd<5>(a, a);
		}

		return a;
	};

	BENCHMARK("mul_binary_tower cpu") {
		for (size_t i = 0; i < NUM_OPS; i++) {
			a = FanPaarTowerField<5>::multiply(a, a);
		}

		return a;
	};
}

TEST_CASE("mul_binary_tower_32b_bitsliced_unrolled", "[mul]") {
	const int TEST_TOWER_HEIGHT = 5;
	uint32_t a[32];
	uint32_t b[32];
	uint32_t result[32];

	for (uint32_t i = 0; i < 32; ++i) {
		result[i] = 0;
	}

	a[0] = 0xd82c07cd;
	b[0] = 0xd82c07cd;

	a[1] = 0x6b4c9946;
	b[1] = 0xd82c07cd;

	a[2] = 0x6b4c9946;
	b[2] = 0x3d47e731;

	a[3] = 0xbe127079;
	b[3] = 0xd82c07cd;

	a[4] = 0xbe127079;
	b[4] = 0x2cd911fc;

	BitsliceUtils<(1 << TEST_TOWER_HEIGHT)>::bitslice_transpose(a);

	BitsliceUtils<(1 << TEST_TOWER_HEIGHT)>::bitslice_transpose(b);

	multiply_unrolled<TEST_TOWER_HEIGHT>(a, b, result);

	BitsliceUtils<(1 << TEST_TOWER_HEIGHT)>::bitslice_untranspose(result);

	REQUIRE(result[0] == 0xafab1b8f);
	REQUIRE(result[1] == 0xf35c8d0f);
	REQUIRE(result[2] == 0xf849322d);
	REQUIRE(result[3] == 0xd86f9eba);
	REQUIRE(result[4] == 0x2b8b8f27);

	uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator(seed);

	for (int i = 0; i < 32; ++i) {
		a[i] = generator();
	}

	BENCHMARK("mul_binary_tower_32b_bitsliced_unrolled cpu") {
		for (size_t i = 0; i < (1 + NUM_OPS / (32 * 3)); i++) {
			multiply_unrolled<TEST_TOWER_HEIGHT>(a, b, result);
			multiply_unrolled<TEST_TOWER_HEIGHT>(b, result, a);
			multiply_unrolled<TEST_TOWER_HEIGHT>(result, a, b);
		}

		return a;
	};
}

TEST_CASE("mul_binary_tower_128b_bitsliced_unrolled", "[mul]") {
	const int TEST_TOWER_HEIGHT = 7;
	uint32_t a[1 << TEST_TOWER_HEIGHT];
	uint32_t b[1 << TEST_TOWER_HEIGHT];
	uint32_t result[1 << TEST_TOWER_HEIGHT];

	for (uint32_t i = 0; i < 4; ++i) {
		result[i] = 0;
	}

	std::string field_elem_a_str = "0xf31223322755a4797859382795323434";

	std::string field_elem_b_str = "0xd3473493847943875934759322048438";

	write_string_to_int_arr(a, field_elem_a_str);

	write_string_to_int_arr(b, field_elem_b_str);

	BitsliceUtils<(1 << TEST_TOWER_HEIGHT)>::bitslice_transpose(a);

	BitsliceUtils<(1 << TEST_TOWER_HEIGHT)>::bitslice_transpose(b);

	multiply_unrolled<TEST_TOWER_HEIGHT>(a, b, result);

	BitsliceUtils<(1 << TEST_TOWER_HEIGHT)>::bitslice_untranspose(result);

	REQUIRE(result[0] == 0x4b3220e5);
	REQUIRE(result[1] == 0x999c424f);
	REQUIRE(result[2] == 0x2dc6d28c);
	REQUIRE(result[3] == 0xceaa247e);

	uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator(seed);

	for (int i = 0; i < (1 << TEST_TOWER_HEIGHT); ++i) {
		a[i] = generator();
	}

	BENCHMARK("mul_binary_tower_128b_bitsliced_unrolled cpu") {
		for (size_t i = 0; i < (1 + NUM_OPS / (32 * 3)); i++) {
			multiply_unrolled<TEST_TOWER_HEIGHT>(a, b, result);
			multiply_unrolled<TEST_TOWER_HEIGHT>(b, result, a);
			multiply_unrolled<TEST_TOWER_HEIGHT>(result, a, b);
		}

		return a;
	};
}

TEST_CASE("mul_bb31", "[mul]") {
	uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();

	std::mt19937 generator(seed);
	BB31 a = BB31(generator());

	BENCHMARK("mul_bb31 cpu") {
		for (size_t i = 0; i < NUM_OPS; i++) {
			a = a * a;
		}
		return a;
	};
}
