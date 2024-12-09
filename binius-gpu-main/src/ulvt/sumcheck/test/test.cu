#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <iostream>
#include <vector>

#include "../../utils/bitslicing.cuh"
#include "../sumcheck.cuh"
#include "../utils/constants.hpp"
#include "./verifier.cuh"
#include "utils/bigints.cuh"

template <uint32_t NUM_VARS, uint32_t COMPOSITION_SIZE, bool DATA_IS_TRANSPOSED>
void test_sumcheck() {
	constexpr uint32_t interpolation_points = COMPOSITION_SIZE + 1;
	const size_t num_ints_in_evals = INTS_PER_VALUE * (1 << NUM_VARS) * COMPOSITION_SIZE;

	std::srand(std::time(nullptr));
	std::vector<uint32_t> multilinear_evals(num_ints_in_evals);

	for (size_t i = 0; i < num_ints_in_evals; ++i) {
		multilinear_evals[i] = std::rand();
	}

	Sumcheck<NUM_VARS, COMPOSITION_SIZE, DATA_IS_TRANSPOSED> s(multilinear_evals, false);

	__uint128_t challenges_bigints[NUM_VARS];

	__uint128_t expected_claim;

	for (uint32_t round = 0; round < NUM_VARS; ++round) {
		std::array<uint32_t, INTS_PER_VALUE> sum;
		std::array<uint32_t, interpolation_points * INTS_PER_VALUE> points;

		s.this_round_messages(sum, points);

		__uint128_t sum_bigint = to_bigint(sum.data());

		// Check that this round's sum matches the next claim expected by the verifier from the last fold

		REQUIRE(((round == 0) || (sum_bigint == expected_claim)));

		__uint128_t points_bigint_arr[interpolation_points];

		to_bigint_arr(points.data(), points_bigint_arr, interpolation_points);

		// Check that this round's sum matches the previous claim expected by the verifier from the current fold

		REQUIRE(sum_bigint == (points_bigint_arr[0] ^ points_bigint_arr[1]));

		std::array<uint32_t, INTS_PER_VALUE> challenge;
		challenge[0] = std::rand();
		challenge[1] = std::rand();
		challenge[2] = std::rand();
		challenge[3] = std::rand();

		__uint128_t challenge_bigint = to_bigint(challenge.data());

		challenges_bigints[round] = challenge_bigint;

		// Set the verifier's expected next claim for the current fold

		expected_claim = evaluate_univariate_given_points(challenge_bigint, points_bigint_arr, interpolation_points);

		s.move_to_next_round(challenge);
	}

	std::array<uint32_t, INTS_PER_VALUE> sum;
	std::array<uint32_t, interpolation_points * INTS_PER_VALUE> points;

	s.this_round_messages(sum, points);

	__uint128_t sum_bigint = to_bigint(sum.data());

	// Check that this round's sum matches the next claim expected by the verifier from the last fold

	REQUIRE(sum_bigint == expected_claim);

	// Check the random evaluation claim by brute force

	// 1. Untranspose all the data if necessary
	if (DATA_IS_TRANSPOSED) {
		uint32_t *gpu_evals;

		cudaMalloc(&gpu_evals, num_ints_in_evals * sizeof(uint32_t));
		cudaMemcpy(gpu_evals, multilinear_evals.data(), num_ints_in_evals * sizeof(uint32_t), cudaMemcpyHostToDevice);

		untranspose_kernel<BITS_WIDTH><<<BLOCKS, THREADS_PER_BLOCK>>>(gpu_evals, num_ints_in_evals / BITS_WIDTH);
		cudaDeviceSynchronize();

		cudaMemcpy(multilinear_evals.data(), gpu_evals, num_ints_in_evals * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		cudaFree(gpu_evals);
	}

	__uint128_t *multilinear_evals_bigints = (__uint128_t *)multilinear_evals.data();

	__uint128_t claimed_evaluation =
		evaluate_multilinear_composition(multilinear_evals_bigints, challenges_bigints, NUM_VARS, COMPOSITION_SIZE);

	REQUIRE(expected_claim == claimed_evaluation);
}

TEST_CASE("sumcheck 20 vars", "[sumcheck]") {
	test_sumcheck<20, 2, true>();

	test_sumcheck<20, 3, true>();

	test_sumcheck<20, 4, true>();

	test_sumcheck<20, 2, false>();

	test_sumcheck<20, 3, false>();

	test_sumcheck<20, 4, false>();
}

TEST_CASE("sumcheck 24 vars", "[sumcheck]") {
	test_sumcheck<24, 2, true>();

	test_sumcheck<24, 3, true>();

	test_sumcheck<24, 4, true>();

	test_sumcheck<24, 2, false>();

	test_sumcheck<24, 3, false>();

	test_sumcheck<24, 4, false>();
}

TEST_CASE("sumcheck 28 vars", "[sumcheck]") {
	test_sumcheck<28, 2, true>();

	test_sumcheck<28, 3, true>();

	test_sumcheck<28, 4, true>();

	test_sumcheck<28, 2, false>();

	test_sumcheck<28, 3, false>();

	test_sumcheck<28, 4, false>();
}