#include "../../utils/constants.hpp"
#include "../utils/tower_7_mul.cuh"
#include "verifier_kernel.cuh"

__global__ void lagrange_basis_eval(
	const __uint128_t *lagrange_basis_evaluations,
	const __uint128_t *challenges_ordered_tuple,
	__uint128_t *evaluation_result,
	const size_t num_challenges,
	const size_t num_eval_points_per_multilinear
) {
	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;  // start the batch index off at the tid

	for (uint32_t basis_eval_idx = tid; basis_eval_idx < num_eval_points_per_multilinear;
		 basis_eval_idx += gridDim.x * blockDim.x) {
		__uint128_t product = lagrange_basis_evaluations[basis_eval_idx];

		int shifted_basis_eval_idx = basis_eval_idx;

		for (int variable_num = 0; variable_num < num_challenges; ++variable_num) {
			if (shifted_basis_eval_idx & 1) {
				product = tower_height_7_mul(product, challenges_ordered_tuple[(num_challenges - 1) - variable_num]);
			} else {
				product =
					tower_height_7_mul(product, challenges_ordered_tuple[(num_challenges - 1) - variable_num] ^ 1);
			}

			shifted_basis_eval_idx >>= 1;
		}

		uint32_t *result_pointer = ((uint32_t *)evaluation_result);
		uint32_t *product_pointer = ((uint32_t *)(&product));

		for (int i = 0; i < INTS_PER_VALUE; ++i) {
			atomicXor(result_pointer + i, product_pointer[i]);
		}
	}
}