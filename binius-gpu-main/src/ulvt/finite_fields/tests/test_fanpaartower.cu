#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "../circuit_generator/unrolled/binary_tower_unrolled.cuh"
#include "../../utils/bitslicing.cuh"
#include "ulvt/finite_fields/binary_tower.cuh"
#include "ulvt/finite_fields/binary_tower_simd.cuh"

TEST_CASE("FanPaarTowerField 16 multiplications", "[mul]") {
	REQUIRE(mul_binary_tower_32b_simd<4>(0x4f4b, 0x4386) == 0x7202);
	REQUIRE(mul_binary_tower_32b_simd<4>(0x2276, 0xc732) == 0x15f8);
	REQUIRE(mul_binary_tower_32b_simd<4>(0x45a6, 0x30fd) == 0x78f1);
	REQUIRE(mul_binary_tower_32b_simd<4>(0xb6c2, 0x80c5) == 0x41e7);
	REQUIRE(mul_binary_tower_32b_simd<4>(0x190f, 0x3ece) == 0x313b);
	REQUIRE(mul_binary_tower_32b_simd<4>(0x556c, 0x4d2) == 0x4e9c);
	REQUIRE(mul_binary_tower_32b_simd<4>(0x3ba, 0x7d6f) == 0x97bc);
	REQUIRE(mul_binary_tower_32b_simd<4>(0x9f1a, 0x5a23) == 0x7cdc);
	REQUIRE(mul_binary_tower_32b_simd<4>(0x33a4, 0xb4bd) == 0xf117);
	REQUIRE(mul_binary_tower_32b_simd<4>(0xf55c, 0x7796) == 0x6f93);
	REQUIRE(mul_binary_tower_32b_simd<4>(0x2593, 0xb435) == 0xbf68);
	REQUIRE(mul_binary_tower_32b_simd<4>(0x3c42, 0x587e) == 0x11f4);
	REQUIRE(mul_binary_tower_32b_simd<4>(0xf797, 0x722c) == 0xa499);
	REQUIRE(mul_binary_tower_32b_simd<4>(0xfdba, 0x8f62) == 0x4d14);
	REQUIRE(mul_binary_tower_32b_simd<4>(0xc92a, 0xee8) == 0xed17);
	REQUIRE(mul_binary_tower_32b_simd<4>(0x944a, 0xad43) == 0x39ee);
	REQUIRE(mul_binary_tower_32b_simd<4>(0x9acb, 0x15df) == 0xc270);
	REQUIRE(mul_binary_tower_32b_simd<4>(0xddb4, 0x8f96) == 0x4d71);
	REQUIRE(mul_binary_tower_32b_simd<4>(0x35c6, 0x4f5c) == 0x1db0);
	REQUIRE(mul_binary_tower_32b_simd<4>(0xf812, 0x7f13) == 0xeb7c);
}

TEST_CASE("FanPaarTowerField 8 multiplications", "[mul]") {
	REQUIRE(mul_binary_tower_32b_simd<3>(0xe0, 0x76) == 0x96);
	REQUIRE(mul_binary_tower_32b_simd<3>(0x1b, 0xa6) == 0xe5);
	REQUIRE(mul_binary_tower_32b_simd<3>(0xd2, 0xdb) == 0x72);
	REQUIRE(mul_binary_tower_32b_simd<3>(0x9a, 0xe) == 0xb2);
	REQUIRE(mul_binary_tower_32b_simd<3>(0x8d, 0xee) == 0xc1);
	REQUIRE(mul_binary_tower_32b_simd<3>(0xc0, 0x33) == 0x68);
	REQUIRE(mul_binary_tower_32b_simd<3>(0x9a, 0x68) == 0xff);
	REQUIRE(mul_binary_tower_32b_simd<3>(0x3, 0xba) == 0x65);
	REQUIRE(mul_binary_tower_32b_simd<3>(0xe0, 0x20) == 0x57);
	REQUIRE(mul_binary_tower_32b_simd<3>(0xf9, 0x84) == 0x77);
	REQUIRE(mul_binary_tower_32b_simd<3>(0x7c, 0x6d) == 0xce);
	REQUIRE(mul_binary_tower_32b_simd<3>(0x5c, 0xb9) == 0x8c);
	REQUIRE(mul_binary_tower_32b_simd<3>(0xa4, 0x48) == 0x38);
	REQUIRE(mul_binary_tower_32b_simd<3>(0x53, 0xb1) == 0x9a);
	REQUIRE(mul_binary_tower_32b_simd<3>(0x70, 0x23) == 0x49);
	REQUIRE(mul_binary_tower_32b_simd<3>(0x83, 0x81) == 0x94);
	REQUIRE(mul_binary_tower_32b_simd<3>(0x40, 0xcb) == 0x77);
	REQUIRE(mul_binary_tower_32b_simd<3>(0xd6, 0xee) == 0x5c);
	REQUIRE(mul_binary_tower_32b_simd<3>(0xdd, 0xc3) == 0x19);
	REQUIRE(mul_binary_tower_32b_simd<3>(0xaf, 0xb4) == 0xe5);
}

TEST_CASE("FanPaarTowerField 32 multiplications", "[mul]") {
	REQUIRE(FanPaarTowerField<5>::multiply(0x8b0fb7a7, 0xcc9b526) == 0x1695a347);
	REQUIRE(FanPaarTowerField<5>::multiply(0x15292d36, 0x96ca6d0c) == 0x6be27e5c);
	REQUIRE(FanPaarTowerField<5>::multiply(0xa510df1d, 0xdc41b407) == 0xa68b93b1);
	REQUIRE(FanPaarTowerField<5>::multiply(0x5a727ae6, 0x545e0fe1) == 0xd1beacf8);
	REQUIRE(FanPaarTowerField<5>::multiply(0xce7254e6, 0x4db30a30) == 0xa7604999);
	REQUIRE(FanPaarTowerField<5>::multiply(0xf81191be, 0xe366f2e) == 0x242a14fb);
	REQUIRE(FanPaarTowerField<5>::multiply(0x7d12a994, 0xe2df7626) == 0x99ccafd0);
	REQUIRE(FanPaarTowerField<5>::multiply(0xf842fb9, 0xc62861bb) == 0xe9c53105);
	REQUIRE(FanPaarTowerField<5>::multiply(0x85bac424, 0xf4ecaf9) == 0x57e5c123);
	REQUIRE(FanPaarTowerField<5>::multiply(0xb2e07978, 0x4b65ff89) == 0x589f6811);
	REQUIRE(FanPaarTowerField<5>::multiply(0x16b4dd34, 0xffb94d84) == 0xc41e546f);
	REQUIRE(FanPaarTowerField<5>::multiply(0xb6638341, 0x56be64f1) == 0x39513551);
	REQUIRE(FanPaarTowerField<5>::multiply(0x6cd7829f, 0x993c39d2) == 0xc2b49a16);
	REQUIRE(FanPaarTowerField<5>::multiply(0x43ee57fe, 0x8f74f10b) == 0xe9327422);
	REQUIRE(FanPaarTowerField<5>::multiply(0xc3a8a8f1, 0x8dd4c194) == 0xa4bd9048);
	REQUIRE(FanPaarTowerField<5>::multiply(0xe5f8605e, 0x53cbc3ac) == 0x3992ec5e);
	REQUIRE(FanPaarTowerField<5>::multiply(0x709bbef, 0xcb2c72bc) == 0x9a14fb2);
	REQUIRE(FanPaarTowerField<5>::multiply(0xf50ab4fe, 0xb9fee15d) == 0xe2bd264e);
}

TEST_CASE("FanPaarTowerField 32 squares", "[sqr]") {
	REQUIRE(FanPaarTowerField<5>::square(0xf8c6fcec) == 0x1e790ce);
	REQUIRE(FanPaarTowerField<5>::square(0xad1dcaf0) == 0x4190653);
	REQUIRE(FanPaarTowerField<5>::square(0xeb94b65) == 0xe3d07a10);
	REQUIRE(FanPaarTowerField<5>::square(0x4232ac3e) == 0xf7cac33e);
	REQUIRE(FanPaarTowerField<5>::square(0xe0089cc2) == 0x4b13d2df);
	REQUIRE(FanPaarTowerField<5>::square(0xe7d35b2) == 0x14d09875);
	REQUIRE(FanPaarTowerField<5>::square(0x68bd9742) == 0xabc65700);
	REQUIRE(FanPaarTowerField<5>::square(0x8a46e227) == 0x5ee5c606);
	REQUIRE(FanPaarTowerField<5>::square(0xa605f25c) == 0x9249ee0f);
	REQUIRE(FanPaarTowerField<5>::square(0x497d342c) == 0x829ac2cd);
	REQUIRE(FanPaarTowerField<5>::square(0x2c1400b9) == 0x2facac56);
	REQUIRE(FanPaarTowerField<5>::square(0xc67e1b8d) == 0x9dff2bce);
	REQUIRE(FanPaarTowerField<5>::square(0xddcc6e06) == 0x722b4d2d);
	REQUIRE(FanPaarTowerField<5>::square(0xff7f8009) == 0xf257f206);
	REQUIRE(FanPaarTowerField<5>::square(0xb7e3728e) == 0xcdddf93);
	REQUIRE(FanPaarTowerField<5>::square(0x64a11278) == 0x14269298);
	REQUIRE(FanPaarTowerField<5>::square(0x52fe395) == 0x2f80b3e6);
	REQUIRE(FanPaarTowerField<5>::square(0x7ea18be8) == 0x6de217db);
	REQUIRE(FanPaarTowerField<5>::square(0x46f5c89) == 0xc6900ed8);
	REQUIRE(FanPaarTowerField<5>::square(0x687c1097) == 0x54c64214);
}

TEST_CASE("FanPaarTowerField 32b inverses", "[inv]") {
	REQUIRE(FanPaarTowerField<5>::inverse(0x1d809f9e) == 0xe731bcf4);
	REQUIRE(FanPaarTowerField<5>::inverse(0x5cd22dea) == 0x1764f442);
	REQUIRE(FanPaarTowerField<5>::inverse(0x359d1eda) == 0x224f1013);
	REQUIRE(FanPaarTowerField<5>::inverse(0x9fb7f3c9) == 0x31043dfe);
	REQUIRE(FanPaarTowerField<5>::inverse(0x7a2052c1) == 0x4d53ce19);
	REQUIRE(FanPaarTowerField<5>::inverse(0x7b0ca83d) == 0xc64879dd);
	REQUIRE(FanPaarTowerField<5>::inverse(0xd8595c69) == 0x218e7b3d);
	REQUIRE(FanPaarTowerField<5>::inverse(0xc5754984) == 0x9c4180d0);
	REQUIRE(FanPaarTowerField<5>::inverse(0x6d58e041) == 0x11d8bf6);
	REQUIRE(FanPaarTowerField<5>::inverse(0x39a5883c) == 0xab677dbe);
	REQUIRE(FanPaarTowerField<5>::inverse(0x67ea2529) == 0x87e784b);
	REQUIRE(FanPaarTowerField<5>::inverse(0xcf61f54c) == 0x5da74a0e);
	REQUIRE(FanPaarTowerField<5>::inverse(0xb4bf2178) == 0x22b84e2b);
	REQUIRE(FanPaarTowerField<5>::inverse(0xe155d245) == 0xa366d524);
	REQUIRE(FanPaarTowerField<5>::inverse(0x9710c57f) == 0xf29cfa4);
	REQUIRE(FanPaarTowerField<5>::inverse(0xce34203c) == 0x927c60e3);
	REQUIRE(FanPaarTowerField<5>::inverse(0x87e15651) == 0x6d4625d1);
	REQUIRE(FanPaarTowerField<5>::inverse(0x2fbd30ed) == 0xf8c6a8d9);
	REQUIRE(FanPaarTowerField<5>::inverse(0xc0a4fe94) == 0xd1115e9);
	REQUIRE(FanPaarTowerField<5>::inverse(0xe77e2c03) == 0x769f80ae);
}

TEST_CASE("Bitsliced 32 multiplications", "[mul]") {
	const int TEST_TOWER_HEIGHT = 5;
	const int WIDTH = 1 << TEST_TOWER_HEIGHT;

	// Initialize 'a' and 'b' with array initializers
	uint32_t a[WIDTH] = {
		0x15292d36,
		0xa510df1d,
		0x5a727ae6,
		0xce7254e6,
		0xf81191be,
		0x7d12a994,
		0x0f842fb9,
		0x85bac424,
		0xb2e07978,
		0x16b4dd34,
		0xb6638341,
		0x6cd7829f,
		0x43ee57fe,
		0xc3a8a8f1,
		0xe5f8605e,
		0x0709bbef,
		0xf50ab4fe
	};

	uint32_t b[WIDTH] = {
		0x96ca6d0c,
		0xdc41b407,
		0x545e0fe1,
		0x4db30a30,
		0x0e366f2e,
		0xe2df7626,
		0xc62861bb,
		0x0f4ecaf9,
		0x4b65ff89,
		0xffb94d84,
		0x56be64f1,
		0x993c39d2,
		0x8f74f10b,
		0x8dd4c194,
		0x53cbc3ac,
		0xcb2c72bc,
		0xb9fee15d
	};

	uint32_t results[WIDTH];

	// Perform bitslice transpose on 'a' and 'b'
	BitsliceUtils<WIDTH>::bitslice_transpose(a);
	BitsliceUtils<WIDTH>::bitslice_transpose(b);

	// Perform the multiplication
	multiply_unrolled<TEST_TOWER_HEIGHT>(a, b, results);

	// Perform bitslice untranspose on 'results'
	BitsliceUtils<WIDTH>::bitslice_untranspose(results);

	// Verify the results
	REQUIRE(results[0] == 0x6be27e5c);
	REQUIRE(results[1] == 0xa68b93b1);
	REQUIRE(results[2] == 0xd1beacf8);
	REQUIRE(results[3] == 0xa7604999);
	REQUIRE(results[4] == 0x242a14fb);
	REQUIRE(results[5] == 0x99ccafd0);
	REQUIRE(results[6] == 0xe9c53105);
	REQUIRE(results[7] == 0x57e5c123);
	REQUIRE(results[8] == 0x589f6811);
	REQUIRE(results[9] == 0xc41e546f);
	REQUIRE(results[10] == 0x39513551);
	REQUIRE(results[11] == 0xc2b49a16);
	REQUIRE(results[12] == 0xe9327422);
	REQUIRE(results[13] == 0xa4bd9048);
	REQUIRE(results[14] == 0x3992ec5e);
	REQUIRE(results[15] == 0x09a14fb2);
	REQUIRE(results[16] == 0xe2bd264e);
}

TEST_CASE("Bitsliced 128-bit multiplications") {
	const int TEST_TOWER_HEIGHT = 7;
	const int WIDTH = 1 << TEST_TOWER_HEIGHT;

	// Initialize operands 'a' and 'b' using little-endian format
	uint32_t a[WIDTH] = {
		0x8f6dc607,
		0xa711bb62,
		0xddc0bb8a,
		0x115035f2,
		0x15420de2,
		0x74be5209,
		0x8bb27903,
		0x9c18dfc4,
		0x765688d4,
		0x92e59968,
		0x0fea136e,
		0x094b6b31,
		0x4e8f5416,
		0xc0dac64f,
		0x01185eb8,
		0xaf6481ec
	};

	uint32_t b[WIDTH] = {
		0x79bf29b4,
		0x8b2e7d96,
		0x95470b21,
		0x03fd844d,
		0x73a9c5be,
		0x60e8ef14,
		0x3cbc3fcb,
		0x53a98758,
		0x84474b46,
		0x82043a0b,
		0x160d630f,
		0xae4e8310,
		0x8a3274bb,
		0x1b6d6538,
		0x2fc07b8a,
		0xb2910e17
	};

	uint32_t results[WIDTH];

	// Perform bitslice transpose on 'a' and 'b'
	BitsliceUtils<WIDTH>::bitslice_transpose(a);
	BitsliceUtils<WIDTH>::bitslice_transpose(b);

	// Perform the multiplication
	multiply_unrolled<TEST_TOWER_HEIGHT>(a, b, results);

	// Perform bitslice untranspose on 'results'
	BitsliceUtils<WIDTH>::bitslice_untranspose(results);

	// Verify the results
	REQUIRE(results[0] == 0x7071882c);
	REQUIRE(results[1] == 0x327aba9c);
	REQUIRE(results[2] == 0xeecf9c0a);
	REQUIRE(results[3] == 0xa096a0f8);

	REQUIRE(results[4] == 0x60015f1b);
	REQUIRE(results[5] == 0xb3fdfe25);
	REQUIRE(results[6] == 0xd3a4ea55);
	REQUIRE(results[7] == 0xa7f1cc39);

	REQUIRE(results[8] == 0x477213e6);
	REQUIRE(results[9] == 0xf80f983d);
	REQUIRE(results[10] == 0x00e27260);
	REQUIRE(results[11] == 0x2601c7b9);

	REQUIRE(results[12] == 0x7295d6a8);
	REQUIRE(results[13] == 0xf3ccbd4e);
	REQUIRE(results[14] == 0x352058ea);
	REQUIRE(results[15] == 0xba61377d);
}
