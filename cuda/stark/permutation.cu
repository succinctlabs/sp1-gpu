#include "interaction.cuh"
#include "../matrix/matrix.cuh"


template<typename F, typename EF> __device__ __forceinline__ EF InteractionValue(
    size_t i, size_t RowIdx, Interactions<F> const interactions,
    Matrix<F> const preprocessed,  Matrix<F> const main, EF const alpha, EF const beta, 
    size_t const batch_size) {
        EF value = EF::zero(); 
        for (size_t j = 0; j < batch_size; j++) {
                // Calculate the interaction index.
                size_t index = i + j;

                if (index >= interactions.num_interactions) {
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

template<typename F, typename EF> __global__ void PopulatePermutationRows(
    Interactions<F> const interactions,
    Matrix<EF> permutation, Matrix<F> const preprocessed, 
    Matrix<F> const main, EF const alpha, EF const beta, size_t const batch_size) {

        size_t RowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (RowIdx >= permutation.height) {
            return;
        }

        EF row_cumulative_sum = EF::zero();
        for (size_t i = 0; i < interactions.num_interactions; i+=batch_size) {
            EF value = InteractionValue(i, RowIdx, interactions, preprocessed, main, alpha, beta, batch_size);
            // Accumulate the sum of values.
            row_cumulative_sum += value;
            // Assign the value to the row.
            size_t perm_index = i / batch_size;
            permutation.values[perm_index * permutation.height + RowIdx] = value;
        }

        // Assign the cumulative sum of values to the last column.
        permutation.values[(permutation.width - 1) * permutation.height + RowIdx] = row_cumulative_sum;
    }


template<typename F, typename EF> __global__ void PopulatePermutationRowsFlattened(
    Interactions<F> const interactions,
    Matrix<F> permutation, Matrix<F> const preprocessed, 
    Matrix<F> const main, EF const alpha, EF const beta, size_t const batch_size) {

        size_t RowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (RowIdx >= permutation.height) {
            return;
        }

        EF row_cumulative_sum = EF::zero();
        for (size_t i = 0; i < interactions.num_interactions; i+=batch_size) {
            EF value = InteractionValue(i, RowIdx, interactions, preprocessed, main, alpha, beta, batch_size);
            // Accumulate the sum of values.
            row_cumulative_sum += value;
            // Assign the value to the row.
            size_t perm_index = (i / batch_size) * EF::D;

            #pragma unroll
            for (size_t k = 0; k < EF::D; k++) {
                size_t flatten_perm_index = perm_index + k;
                permutation.values[flatten_perm_index * permutation.height + RowIdx] = value.value[k];
            }
        }

        // Assign the cumulative sum of values to the last column.
        size_t last_col_index = permutation.width - EF::D;
        #pragma unroll
        for (size_t k = 0; k < EF::D; k++) {
            size_t flatten_perm_index = last_col_index + k;
            permutation.values[flatten_perm_index * permutation.height + RowIdx] = row_cumulative_sum.value[k];
        }
    }

