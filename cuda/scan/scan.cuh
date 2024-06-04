#pragma once


#include "kernels.cu"
#include "../utils/exception.cuh"

#include "../fields/bb31_extension_t.cuh"

template<typename T> RustCudaError scan_template(T * d_in, T * d_out, size_t n) {
    if((2 * n) <= SECTION_SIZE)
        single_block_scan<<<1, n>>>(d_out, d_in, n);
    else {
       size_t block_dim = 512;
       size_t num_blocks = ceil(n / (float)block_dim);
       T * scan_values;
       unsigned int * BlockCounter;
       unsigned int * flags;
       size_t flag_size = sizeof(unsigned int) * (num_blocks + 1);
       CUDA_OK(cudaMalloc(&scan_values, sizeof(T) * (num_blocks + 1)));
       CUDA_OK(cudaMemset(scan_values, 0, sizeof(T)));
       CUDA_OK(cudaMalloc(&BlockCounter, sizeof(unsigned int)));
       CUDA_OK(cudaMemset(BlockCounter, 0, sizeof(unsigned int)));
       CUDA_OK(cudaMalloc(&flags, flag_size));
       CUDA_OK(cudaMemset(flags, 0, flag_size));
       CUDA_OK(cudaMemset(flags, 1, sizeof(unsigned int)));
       scan<<<num_blocks, block_dim>>>(d_out, d_in, n, scan_values, BlockCounter, flags);
       CUDA_OK(cudaFree(BlockCounter));
       CUDA_OK(cudaFree(flags));
    }
    return CUDA_SUCCESS_MOON;
}


extern "C" RustCudaError scan_baby_bear(bb31_t * d_in, bb31_t * d_out, size_t n) {
    return scan_template(d_in, d_out, n);
}

extern "C" RustCudaError scan_baby_bear_challenge(bb31_extension_t * d_in, bb31_extension_t * d_out, size_t n) {
    return scan_template(d_in, d_out, n);
}