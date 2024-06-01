#pragma once

#include "permutation.cu"


extern "C" void populate_permutation_rows(Interactions<bb31_t> const interactions,
                    Matrix<bb31_extension_t> permutation, Matrix<bb31_t> const preprocessed, 
                    Matrix<bb31_t> const main, bb31_extension_t alpha, bb31_extension_t beta, 
                    size_t batch_size, size_t nBlocks, size_t nThreadsPerBlock) {
                        populate_permutation_rows<<<nBlocks, nThreadsPerBlock>>>(
                            interactions, permutation, preprocessed, main, alpha, beta, batch_size);
                    }