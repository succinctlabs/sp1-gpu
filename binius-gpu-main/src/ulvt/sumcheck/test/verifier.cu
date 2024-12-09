#include <iostream>

#include "../utils/constants.hpp"
#include "kernel/verifier_kernel.cuh"
#include "utils/tower_7_mul.cuh"
#include "utils/unbitsliced_mul.cuh"
#include "verifier.cuh"

__uint128_t evaluate_univariate_given_points(
	const __uint128_t challenge, __uint128_t *interpolation_points, const uint32_t num_points
) {
	__uint128_t evaluation = 0;

	// compute each of the interpolation products
	for (int current_point_idx = 0; current_point_idx < num_points; ++current_point_idx) {
		__uint128_t interpolation_product = interpolation_points[current_point_idx];

		for (int other_point_idx = 0; other_point_idx < num_points; ++other_point_idx) {
			if (other_point_idx != current_point_idx) {
				interpolation_product = tower_height_7_mul(interpolation_product, challenge ^ other_point_idx);
				interpolation_product = tower_height_7_mul(
					interpolation_product, inverse_at_interpolation_point(current_point_idx ^ other_point_idx)
				);
			}
		}

		evaluation ^= interpolation_product;
	}

	return evaluation;
}

__uint128_t evaluate_multilinear_given_point(
	__uint128_t *lagrange_basis_evaluations, __uint128_t *challenges_ordered_tuple, const size_t num_challenges
) {
	const size_t num_basis_evals = 1 << num_challenges;

	__uint128_t evaluation = 0;

	__uint128_t *gpu_lagrange_basis_evaluations, *gpu_challenges_ordered_tuple, *gpu_eval;

	cudaMalloc(&gpu_lagrange_basis_evaluations, num_basis_evals * sizeof(__uint128_t));
	cudaMalloc(&gpu_eval, sizeof(__uint128_t));
	cudaMemset(gpu_eval, 0, sizeof(__uint128_t));
	cudaMalloc(&gpu_challenges_ordered_tuple, num_challenges * sizeof(__uint128_t));

	cudaMemcpy(
		gpu_lagrange_basis_evaluations,
		lagrange_basis_evaluations,
		num_basis_evals * sizeof(__uint128_t),
		cudaMemcpyHostToDevice
	);

	cudaMemcpy(
		gpu_challenges_ordered_tuple,
		challenges_ordered_tuple,
		num_challenges * sizeof(__uint128_t),
		cudaMemcpyHostToDevice
	);

	lagrange_basis_eval<<<256, 256>>>(
		gpu_lagrange_basis_evaluations, gpu_challenges_ordered_tuple, gpu_eval, num_challenges, num_basis_evals
	);

	cudaDeviceSynchronize();

	cudaMemcpy(
		lagrange_basis_evaluations,
		gpu_lagrange_basis_evaluations,
		num_basis_evals * sizeof(__uint128_t),
		cudaMemcpyDeviceToHost
	);
	cudaMemcpy(&evaluation, gpu_eval, sizeof(__uint128_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(
		challenges_ordered_tuple,
		gpu_challenges_ordered_tuple,
		num_challenges * sizeof(__uint128_t),
		cudaMemcpyDeviceToHost
	);

	cudaFree(gpu_lagrange_basis_evaluations);
	cudaFree(gpu_eval);
	cudaFree(gpu_challenges_ordered_tuple);

	return evaluation;
}

__uint128_t evaluate_multilinear_composition(
	__uint128_t *lagrange_basis_evaluations,
	__uint128_t *challenges_ordered_tuple,
	const size_t num_challenges,
	const size_t num_columns
) {
	size_t evals_per_column = 1 << num_challenges;
	__uint128_t product = 1;

	for (int column_num = 0; column_num < num_columns; ++column_num) {
		product = tower_height_7_mul(
			product,
			evaluate_multilinear_given_point(
				lagrange_basis_evaluations + column_num * evals_per_column, challenges_ordered_tuple, num_challenges
			)
		);
	}

	return product;
}
