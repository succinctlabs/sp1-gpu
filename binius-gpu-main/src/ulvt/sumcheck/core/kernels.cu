#include <stdio.h>

#include <cstdint>

#include "../../finite_fields/circuit_generator/unrolled/binary_tower_unrolled.cuh"
#include "../utils/constants.hpp"
#include "core.cuh"

__global__ void fold_large_list_halves(
	uint32_t* source,
	uint32_t* destination,
	uint32_t coefficient[BITS_WIDTH],
	const uint32_t num_batch_rows,
	const uint32_t src_evals_per_column,
	const uint32_t dst_evals_per_column,
	const uint32_t num_cols
) {
	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;  // start the batch index off at the tid

	for (uint32_t row_idx = tid; row_idx < num_batch_rows; row_idx += gridDim.x * blockDim.x) {
		for (uint32_t col_idx = 0; col_idx < num_cols; ++col_idx) {
			uint32_t* lower_batch = source + BITS_WIDTH * row_idx + INTS_PER_VALUE * src_evals_per_column * col_idx;
			uint32_t* upper_batch = lower_batch + BITS_WIDTH * num_batch_rows;

			uint32_t* dst_batch = destination + BITS_WIDTH * row_idx + INTS_PER_VALUE * dst_evals_per_column * col_idx;

			fold_batch(lower_batch, upper_batch, dst_batch, coefficient, false);
		}
	}
}
