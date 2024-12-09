#include <cstdint>

#include "../utils/constants.hpp"

template <uint32_t INTERPOLATION_POINTS, uint32_t COMPOSITION_SIZE, uint32_t EVALS_PER_MULTILINEAR>
__global__ void compute_compositions(
	const uint32_t* multilinear_evaluations,
	uint32_t* multilinear_products_sums,
	uint32_t* folded_products_sums,
	const uint32_t coefficients[INTERPOLATION_POINTS * BITS_WIDTH],
	const uint32_t num_batch_rows,
	const uint32_t active_threads,
	const uint32_t active_threads_folded
) {
	const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;  // start the batch index off at the tid

	uint32_t folded_products_sums_this_thread[INTERPOLATION_POINTS * BITS_WIDTH];

	uint32_t multilinear_products_sums_this_thread[BITS_WIDTH];

	memset(folded_products_sums_this_thread, 0, INTERPOLATION_POINTS * BITS_WIDTH * sizeof(uint32_t));

	memset(multilinear_products_sums_this_thread, 0, BITS_WIDTH * sizeof(uint32_t));

	for (uint32_t row_idx = tid; row_idx < num_batch_rows; row_idx += gridDim.x * blockDim.x) {
		uint32_t this_multilinear_product[BITS_WIDTH];

		evaluate_composition_on_batch_row(
			multilinear_evaluations + BITS_WIDTH * row_idx,
			this_multilinear_product,
			COMPOSITION_SIZE,
			EVALS_PER_MULTILINEAR
		);

		for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
			multilinear_products_sums_this_thread[i] ^= this_multilinear_product[i];
		}

		uint32_t num_batch_rows_to_fold = num_batch_rows / 2;

		if (row_idx < num_batch_rows_to_fold) {
			// Fold each batch in the batch row
			uint32_t folded_batch_row[INTERPOLATION_POINTS * COMPOSITION_SIZE * BITS_WIDTH];

			// Fold this batch with the corresponding one
			for (int column_idx = 0; column_idx < COMPOSITION_SIZE; ++column_idx) {
				uint32_t batches_fitting_into_original_column = EVALS_PER_MULTILINEAR / 32;
				const uint32_t* lower_batch =
					multilinear_evaluations +
					BITS_WIDTH * (batches_fitting_into_original_column * column_idx + row_idx);
				const uint32_t* upper_batch = lower_batch + BITS_WIDTH * num_batch_rows_to_fold;

				for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
					fold_batch(
						lower_batch,
						upper_batch,
						folded_batch_row + BITS_WIDTH * (column_idx * INTERPOLATION_POINTS + interpolation_point),
						coefficients + BITS_WIDTH * interpolation_point,
						true
					);
				}
			}

			// Take the folded batches and evaluate the compositions on them

			for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
				uint32_t this_interpolation_point_product_batch[BITS_WIDTH];
				evaluate_composition_on_batch_row(
					folded_batch_row + BITS_WIDTH * interpolation_point,
					this_interpolation_point_product_batch,
					COMPOSITION_SIZE,
					INTERPOLATION_POINTS * 32
				);

				// Add this product to the sum of all products taken by the thread
				uint32_t* this_interpolation_point_sum_location =
					folded_products_sums_this_thread + BITS_WIDTH * interpolation_point;

				for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
					this_interpolation_point_sum_location[i] ^= this_interpolation_point_product_batch[i];
				}
			}
		}
	}

	if (tid < active_threads) {
		for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
			atomicXor(multilinear_products_sums + i, multilinear_products_sums_this_thread[i]);
		}
	}

	if (tid < active_threads_folded) {
		for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
			uint32_t* batch_to_copy_to = folded_products_sums + BITS_WIDTH * interpolation_point;
			uint32_t* batch_to_copy_from = folded_products_sums_this_thread + BITS_WIDTH * interpolation_point;

			for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
				atomicXor(batch_to_copy_to + i, batch_to_copy_from[i]);
			}
		}
	}
}

__global__ void fold_large_list_halves(
	uint32_t* source,
	uint32_t* destination,
	uint32_t coefficient[BITS_WIDTH],
	const uint32_t num_batch_rows,
	const uint32_t src_evals_per_column,
	const uint32_t dst_evals_per_column,
	const uint32_t num_cols
);