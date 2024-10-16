#include "interaction.cuh"
#include "../matrix/matrix.cuh"


template<typename F, typename EF> __device__ __forceinline__ EF InteractionValue(
    size_t i, size_t RowIdx, Interactions<F> const interactions,
    Matrix<F> const preprocessed,  Matrix<F> const main, EF const alpha, EF const beta,
    size_t const batch_size, size_t const max_idx) {
        EF value = EF::zero(); 
        for (size_t j = 0; j < batch_size; j++) {
                // Calculate the interaction index.
                size_t index = i + j;

                if (index >= max_idx) {
                    break;
                }

                // Initialize the denominator and beta powers.
                EF denominator = alpha;
                EF beta_power = EF::one();

                // Add argument index to the denominator.
                EF argument_index = EF(interactions.arg_indices[index]);
                denominator += beta_power * argument_index;

                // Add the interaction values.
                for (size_t k = interactions.values_ptr[index]; k < interactions.values_ptr[index + 1]; k++) {
                    beta_power *= beta;
                    EF acc = EF(interactions.values_constants[k]);
                    for (size_t l = interactions.values_col_weights_ptr[k]; l < interactions.values_col_weights_ptr[k + 1]; l++) {
                        acc += EF(interactions.values_col_weights[l].get(preprocessed, main, RowIdx));
                    }
                    denominator += beta_power * acc;
                }

                // Calculate the multiplicity values.
                bool is_send = interactions.is_send[index];
                F mult = interactions.mult_constants[index];

                for (size_t k = interactions.multiplicities_ptr[index]; k < interactions.multiplicities_ptr[index + 1]; k++) {
                    mult += interactions.mult_col_weights[k].get(preprocessed, main, RowIdx);
                }

                if (!is_send) {
                    mult = F(0) - mult;
                }

                // Add `mult/ denominator` to the sum.
                value += EF(mult) / denominator;
            }

            return value;
    }

template<typename F, typename EF> __global__ void PopulatePermutationRowsFlattened(
    Interactions<F> const interactions,
    Matrix<F> permutation, Matrix<F> const preprocessed, 
    Matrix<F> const main, EF const global_alpha, EF const global_beta, EF const local_alpha,
    EF const local_beta, size_t const batch_size) {
        size_t RowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (RowIdx >= permutation.height) {
            return;
        }

        size_t num_interactions = interactions.num_global_interactions + interactions.num_local_interactions;

        // Within `interactions`, the global interactions are first, then the local ones.
        EF cumulative_sum = EF::zero();
        for (size_t i = 0, perm_ext_element_idx = 0; i < num_interactions; )
        {
            bool is_global = i < interactions.num_global_interactions;
            size_t max_scope_inter_idx = is_global ? interactions.num_global_interactions : num_interactions;

            // Calculate the batched interaction value.
            EF alpha = is_global ? global_alpha : local_alpha;
            EF beta = is_global ? global_beta : local_beta;
            EF value = InteractionValue(i, RowIdx, interactions, preprocessed, main, alpha, beta, batch_size, max_scope_inter_idx);

            // Assign the value to the extension field slot in the permutation trace.
            size_t perm_start_idx = perm_ext_element_idx * EF::D;
            #pragma unroll
            for (size_t k = 0; k < EF::D; k++) {
                size_t flatten_perm_idx = perm_start_idx + k;
                permutation.values[flatten_perm_idx * permutation.height + RowIdx] = value.value[k];
            }
            perm_ext_element_idx++;

            // Accumulate the sum of values for the scope.
            cumulative_sum += value;

            // If we are in the last batch, assign the cumulative sum.
            bool is_last_batch = i + batch_size >= max_scope_inter_idx;
            if (is_last_batch) {
                size_t cumulaitve_sum_start_idx = perm_ext_element_idx * EF::D;
                #pragma unroll
                for (size_t k = 0; k < EF::D; k++) {
                    size_t flatten_perm_idx = cumulaitve_sum_start_idx + k;
                    permutation.values[flatten_perm_idx * permutation.height + RowIdx] = cumulative_sum.value[k];
                }
                perm_ext_element_idx++;
                cumulative_sum = EF::zero();
            }

            // Calculate the current batch size to increment i.
            size_t current_batch_size = min(batch_size, max_scope_inter_idx - i);
            i += current_batch_size;
        }
    }

