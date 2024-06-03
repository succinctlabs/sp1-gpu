#pragma once

#include "permutation.cu"


extern "C" void populate_permutation_rows(Interactions<bb31_t> interactions,
                    Matrix<bb31_extension_t> permutation, Matrix<bb31_t> preprocessed, 
                    Matrix<bb31_t> main, bb31_extension_t alpha, bb31_extension_t beta, 
                    size_t batch_size, size_t nBlocks, size_t nThreadsPerBlock) {
                        assert(!permutation.row_major);
                        assert(!main.row_major);
                        assert(!preprocessed.row_major);
                        populate_permutation_rows<<<nBlocks, nThreadsPerBlock>>>(
                            interactions, permutation, preprocessed, main, alpha, beta, batch_size);
                    }