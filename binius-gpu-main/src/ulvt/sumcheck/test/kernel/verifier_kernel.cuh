#include <cstdint>

__global__ void lagrange_basis_eval(
	const __uint128_t *lagrange_basis_evaluations,
	const __uint128_t *challenges_ordered_tuple,
	__uint128_t *products,
	const size_t num_challenges,
	const size_t num_eval_points_per_multilinear
);