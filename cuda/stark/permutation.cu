

#include "interaction.cuh"
#include "../utils/matrix.cuh"

template<typename F, typename EF> __global__ void populate_permutation_rows(
    Interactions<F> const interactions,
    Matrix<EF> permutation, Matrix<F> const preprocessed, 
    Matrix<F> const main, EF alpha, EF beta, size_t batch_size) {

        size_t RowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

        F *main_row = main.values + RowIdx * main.width;
        F *prep_row = preprocessed.values + RowIdx * preprocessed.width;
        EF *perm_row = permutation.values + RowIdx * permutation.width;

        for (size_t i = 0; i < interactions.num_interactions; i+=batch_size) {
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
                    EF acc = interactions.values_constants[k];
                    for (size_t l = interactions.values_col_weights_ptr[k]; l < interactions.values_col_weights_ptr[k + 1]; l++) {
                        acc += interactions.values_col_weights[l].get(prep_row, main_row);
                    }
                    denominator += beta_power * acc;
                }

                // Calculate the multiplicity values.
                bool is_send = interactions.is_send[index];
                EF mult = interactions.mult_constants[index];

                for (size_t k = interactions.multiplicities_ptr[index]; k < interactions.multiplicities_ptr[index + 1]; k++) {
                    mult += interactions.mult_col_weights[k].get(prep_row, main_row);
                }

                if (!is_send) {
                    mult = EF::zero() -mult;
                }

                // Add `mult/ denominator` to the sum.
                value += EF(mult) / denominator;
            }
            // Assign the value to the row.
            size_t row_index = i / batch_size;
            permutation.values[row_index] = value;
        }

    };

