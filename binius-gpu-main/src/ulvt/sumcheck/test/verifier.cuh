#include <iostream>

__uint128_t evaluate_univariate_given_points(
	__uint128_t challenge, __uint128_t *interpolation_points, uint32_t num_points
);

__uint128_t evaluate_multilinear_composition(
	__uint128_t *lagrange_basis_evaluations,
	__uint128_t *challenges_ordered_tuple,
	size_t num_challenges,
	size_t num_columns
);