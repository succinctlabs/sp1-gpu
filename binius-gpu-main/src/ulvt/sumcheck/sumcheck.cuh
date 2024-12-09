#include <array>
#include <chrono>
#include <vector>

#include "../utils/bitslicing.cuh"
#include "core/core.cuh"
#include "core/kernels.cuh"
#include "utils/constants.hpp"

template <uint32_t NUM_VARS, uint32_t COMPOSITION_SIZE, bool DATA_IS_TRANSPOSED>
class Sumcheck {
	static_assert(NUM_VARS == 20 || NUM_VARS == 24 || NUM_VARS == 28, "NUM_VARS must be 20, 24, or 28");
	static_assert(
		COMPOSITION_SIZE == 2 || COMPOSITION_SIZE == 3 || COMPOSITION_SIZE == 4, "COMPOSITION_SIZE must be 2, 3, or 4"
	);

private:
	static constexpr uint32_t INTERPOLATION_POINTS = COMPOSITION_SIZE + 1;

	static constexpr uint32_t EVALS_PER_MULTILINEAR = 1 << NUM_VARS;

	static constexpr size_t TOTAL_INTS = INTS_PER_VALUE * EVALS_PER_MULTILINEAR * COMPOSITION_SIZE;
	uint32_t coefficients[BITS_WIDTH * INTERPOLATION_POINTS];

	uint32_t round = 0;

	uint32_t *cpu_multilinear_evaluations, *gpu_multilinear_evaluations;

	uint32_t *gpu_coefficients;

	void fold_list_halves(
		uint32_t *source,
		uint32_t *destination,
		uint32_t *coefficient,
		const size_t list_len,
		const uint32_t src_original_evals_per_column,
		const uint32_t dst_original_evals_per_column,
		const uint32_t num_cols
	) {
		size_t batches_in_half_list = list_len >> 6;

		if (batches_in_half_list > 0) {
			// For lists of size >32 elements, fold in half with folding factor
			// Assume source lives on the GPU
			uint32_t *gpu_coefficient;

			cudaMalloc(&gpu_coefficient, sizeof(uint32_t) * BITS_WIDTH);

			cudaMemcpy(gpu_coefficient, coefficient, sizeof(uint32_t) * BITS_WIDTH, cudaMemcpyHostToDevice);

			fold_large_list_halves<<<BLOCKS, THREADS_PER_BLOCK>>>(
				source,
				destination,
				gpu_coefficient,
				batches_in_half_list,
				src_original_evals_per_column,
				dst_original_evals_per_column,
				num_cols
			);

			cudaDeviceSynchronize();
		} else {
			for (uint32_t col_idx = 0; col_idx < num_cols; ++col_idx) {
				// For small lists, copy over the later half into top of new batch, multiply by r, and add
				// Assume source lives on the CPU

				uint32_t *this_col_src = source + BITS_WIDTH * col_idx;
				uint32_t *this_col_dst = destination + BITS_WIDTH * col_idx;

				fold_small(this_col_src, this_col_dst, coefficient, list_len);
			}
		}
	}

public:
	std::chrono::time_point<std::chrono::high_resolution_clock> start_before_memcpy;

	std::chrono::time_point<std::chrono::high_resolution_clock> start_before_transpose;

	std::chrono::time_point<std::chrono::high_resolution_clock> start_raw;

	Sumcheck(const std::vector<uint32_t> &evals_span, const bool benchmarking) {
		const uint32_t *evals = evals_span.data();
		cpu_multilinear_evaluations = new uint32_t[BITS_WIDTH * COMPOSITION_SIZE];
		cudaMalloc(&gpu_multilinear_evaluations, sizeof(uint32_t) * TOTAL_INTS);

		if (benchmarking) {
			start_before_memcpy = std::chrono::high_resolution_clock::now();
		}

		cudaMemcpy(gpu_multilinear_evaluations, evals, sizeof(uint32_t) * TOTAL_INTS, cudaMemcpyHostToDevice);

		if (benchmarking) {
			start_before_transpose = std::chrono::high_resolution_clock::now();
		}

		if (!DATA_IS_TRANSPOSED) {
			transpose_kernel<BITS_WIDTH>
				<<<BLOCKS, THREADS_PER_BLOCK>>>(gpu_multilinear_evaluations, TOTAL_INTS / BITS_WIDTH);
			cudaDeviceSynchronize();
		}

		for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
			uint32_t coefficient_as_value[INTS_PER_VALUE];

			coefficient_as_value[0] = interpolation_point;

			for (int i = 1; i < INTS_PER_VALUE; ++i) {
				coefficient_as_value[i] = 0;
			}

			BitsliceUtils<BITS_WIDTH>::repeat_value_bitsliced(
				coefficients + interpolation_point * BITS_WIDTH, coefficient_as_value
			);
		}

		cudaMalloc(&gpu_coefficients, BITS_WIDTH * INTERPOLATION_POINTS * sizeof(uint32_t));

		cudaMemcpy(
			gpu_coefficients, coefficients, BITS_WIDTH * INTERPOLATION_POINTS * sizeof(uint32_t), cudaMemcpyHostToDevice
		);

		if (benchmarking) {
			start_raw = std::chrono::high_resolution_clock::now();
		}
	}

	~Sumcheck() { delete[] cpu_multilinear_evaluations; }

	void this_round_messages(
		std::array<uint32_t, INTS_PER_VALUE> &sum_span,
		std::array<uint32_t, INTERPOLATION_POINTS * INTS_PER_VALUE> &points_span
	) {
		uint32_t *sum = sum_span.data();
		uint32_t *points = points_span.data();

		// Sum is evaluated by taking batches of 32 and summing them up to 1 batch
		uint32_t num_eval_points_per_multilinear_unpadded = EVALS_PER_MULTILINEAR >> round;
		uint32_t num_eval_points_per_multilinear_padded = std::max(32u, num_eval_points_per_multilinear_unpadded);

		uint32_t num_eval_points_per_folded_multilinear_unpadded = num_eval_points_per_multilinear_unpadded / 2;

		uint32_t num_eval_points_per_folded_multilinear_padded =
			std::max(32u, num_eval_points_per_folded_multilinear_unpadded);

		const uint32_t num_batches_per_multilinear = num_eval_points_per_multilinear_padded / 32;

		const uint32_t num_batches_per_folded_multilinear = num_eval_points_per_multilinear_padded / 64;

		const uint32_t active_threads = std::min(num_batches_per_multilinear, THREADS);

		const uint32_t active_threads_folded = std::min(num_batches_per_folded_multilinear, THREADS);

		uint32_t multilinear_products[BITS_WIDTH];

		uint32_t *folded_products_sums = new uint32_t[INTERPOLATION_POINTS * BITS_WIDTH];

		uint32_t *gpu_multilinear_products, *gpu_folded_products_sums;

		if (num_eval_points_per_multilinear_padded == 32) {
			// If the number of evals fits in a single batch, use the CPU

			// 1. Calculate the products of the multilinear evaluations
			evaluate_composition_on_batch_row(cpu_multilinear_evaluations, multilinear_products, COMPOSITION_SIZE, 32);

			// For each interpolation point, fold according to that point, and load the result into "folded_at_point"

			for (uint32_t interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
				uint32_t folded_at_point[BITS_WIDTH * COMPOSITION_SIZE] = {0};

				fold_list_halves(
					cpu_multilinear_evaluations,
					folded_at_point,
					coefficients + BITS_WIDTH * interpolation_point,
					num_eval_points_per_multilinear_unpadded,
					32,
					32,
					COMPOSITION_SIZE
				);

				evaluate_composition_on_batch_row(
					folded_at_point, folded_products_sums + (BITS_WIDTH * interpolation_point), COMPOSITION_SIZE, 32
				);
			}

			compute_sum(sum, multilinear_products, num_eval_points_per_multilinear_unpadded);

			for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
				compute_sum(
					points + interpolation_point * INTS_PER_VALUE,
					folded_products_sums +
						num_eval_points_per_folded_multilinear_padded * INTS_PER_VALUE * interpolation_point,
					num_eval_points_per_folded_multilinear_unpadded
				);
			}
		} else {
			cudaMalloc(&gpu_multilinear_products, BITS_WIDTH * sizeof(uint32_t));

			cudaMemset(gpu_multilinear_products, 0, BITS_WIDTH * sizeof(uint32_t));

			cudaMalloc(&gpu_folded_products_sums, INTERPOLATION_POINTS * BITS_WIDTH * sizeof(uint32_t));

			cudaMemset(gpu_folded_products_sums, 0, INTERPOLATION_POINTS * BITS_WIDTH * sizeof(uint32_t));

			compute_compositions<INTERPOLATION_POINTS, COMPOSITION_SIZE, EVALS_PER_MULTILINEAR>
				<<<BLOCKS, THREADS_PER_BLOCK>>>(
					gpu_multilinear_evaluations,
					gpu_multilinear_products,
					gpu_folded_products_sums,
					gpu_coefficients,
					num_batches_per_multilinear,
					active_threads,
					active_threads_folded
				);

			cudaDeviceSynchronize();

			cudaMemcpy(
				multilinear_products, gpu_multilinear_products, BITS_WIDTH * sizeof(uint32_t), cudaMemcpyDeviceToHost
			);

			cudaMemcpy(
				folded_products_sums,
				gpu_folded_products_sums,
				INTERPOLATION_POINTS * BITS_WIDTH * sizeof(uint32_t),
				cudaMemcpyDeviceToHost
			);

			cudaDeviceSynchronize();

			cudaMemcpy(
				multilinear_products, gpu_multilinear_products, BITS_WIDTH * sizeof(uint32_t), cudaMemcpyDeviceToHost
			);

			cudaFree(gpu_multilinear_products);
			cudaFree(gpu_folded_products_sums);

			compute_sum(sum, multilinear_products, 32);

			for (int interpolation_point = 0; interpolation_point < INTERPOLATION_POINTS; ++interpolation_point) {
				uint32_t *point = points + interpolation_point * INTS_PER_VALUE;

				compute_sum(point, folded_products_sums + BITS_WIDTH * interpolation_point, 32);
			}
		}
	};

	void move_to_next_round(const std::array<uint32_t, INTS_PER_VALUE> &challenge_span) {
		const uint32_t *challenge = challenge_span.data();

		// Take a_i(x_i,...,x_n) and create a_(i+1)(x_(i+1),...,x_n) = a_i(challenge,x_(i+1),...,x_n)
		uint32_t coefficient[BITS_WIDTH];

		uint32_t num_eval_points_per_multilinear = EVALS_PER_MULTILINEAR >> round;

		BitsliceUtils<BITS_WIDTH>::repeat_value_bitsliced(coefficient, challenge);

		// Load the folded columns
		if (num_eval_points_per_multilinear <= 32) {
			fold_list_halves(
				cpu_multilinear_evaluations,
				cpu_multilinear_evaluations,
				coefficient,
				num_eval_points_per_multilinear,
				32,
				32,
				COMPOSITION_SIZE
			);
		} else {
			fold_list_halves(
				gpu_multilinear_evaluations,
				gpu_multilinear_evaluations,
				coefficient,
				num_eval_points_per_multilinear,
				EVALS_PER_MULTILINEAR,
				EVALS_PER_MULTILINEAR,
				COMPOSITION_SIZE
			);
		}

		uint32_t new_num_evals_per_multilinear = num_eval_points_per_multilinear / 2;

		if (new_num_evals_per_multilinear == 32) {
			// Now we use cpu to store the evaluations instead of gpu
			for (int column_idx = 0; column_idx < COMPOSITION_SIZE; ++column_idx) {
				cudaMemcpy(
					cpu_multilinear_evaluations + column_idx * BITS_WIDTH,
					gpu_multilinear_evaluations + column_idx * (EVALS_PER_MULTILINEAR * INTS_PER_VALUE),
					sizeof(uint32_t) * BITS_WIDTH,
					cudaMemcpyDeviceToHost
				);
			}

			cudaDeviceSynchronize();

			cudaFree(gpu_multilinear_evaluations);
		}

		++round;
	};
};