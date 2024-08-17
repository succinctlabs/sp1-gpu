// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <sstream>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

struct RustCudaError {
    const char *message;
};

typedef struct RustCudaError rustCudaError_t;

extern "C" const rustCudaError_t CUDA_SUCCESS_MOON =
rustCudaError_t{message : cudaGetErrorString(cudaSuccess)};

extern "C" const rustCudaError_t CUDA_OUT_OF_MEMORY =
rustCudaError_t{message : cudaGetErrorString(cudaErrorMemoryAllocation)};

#define CUDA_UNWRAP(expr)                                             \
    do {                                                              \
        cudaError_t code = expr;                                      \
        if (code != cudaSuccess) {                                    \
            std::stringstream ss;                                     \
            ss << __FILE__ << "(" << __LINE__ << ")";                 \
            std::string file_and_line;                                \
            ss >> file_and_line;                                      \
            throw thrust::system_error(code, thrust::cuda_category(), \
                                       file_and_line);                \
        }                                                             \
    } while (0)

#define CUDA_OK(expr)                                                   \
    do {                                                                \
        cudaError_t code = expr;                                        \
        if (code != cudaSuccess) {                                      \
            return rustCudaError_t{message : cudaGetErrorString(code)}; \
        }                                                               \
    } while (0)
