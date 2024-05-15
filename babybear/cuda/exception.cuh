// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <sstream>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#define CUDA_OK(expr)                                                 \
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
