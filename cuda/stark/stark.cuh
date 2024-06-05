#pragma once

#include "permutation.cu"


extern "C" void populate_permutation_rows(Interactions<bb31_t> const interactions,
                    Matrix<bb31_extension_t> permutation, Matrix<bb31_t> const preprocessed, 
                    Matrix<bb31_t> const main, bb31_extension_t const alpha, bb31_extension_t const beta, 
                    size_t const batch_size, size_t const nBlocks, size_t const nThreadsPerBlock) {
                        assert(!permutation.row_major);
                        assert(!main.row_major);
                        assert(!preprocessed.row_major);
                        PopulatePermutationRows<<<nBlocks, nThreadsPerBlock>>>(
                            interactions, permutation, preprocessed, main, alpha, beta, batch_size);
                    }

extern "C" void populate_permutation_rows_flattened(Interactions<bb31_t> const interactions,
                    Matrix<bb31_t> permutation, Matrix<bb31_t> const preprocessed, 
                    Matrix<bb31_t> const main, bb31_extension_t const alpha, bb31_extension_t const beta, 
                    size_t const batch_size, size_t const nBlocks, size_t const nThreadsPerBlock) {
                        assert(!permutation.row_major);
                        assert(!main.row_major);
                        assert(!preprocessed.row_major);
                        assert(permutation.height == main.height);
                        PopulatePermutationRowsFlattened<<<nBlocks, nThreadsPerBlock>>>(
                            interactions, permutation, preprocessed, main, alpha, beta, batch_size);
                    }