#pragma once

#include "kernels.cu"
#include "../utils/exception.cuh"

#include "../fields/bb31_extension_t.cuh"

template<typename T> RustCudaError ScanTemplate(T * d_out, T * d_in, size_t n, cudaStream_t stream) {
    if((2 * n) <= scan_kernels::SECTION_SIZE)
        scan_kernels::SingleBlockScan<<<1, n, 0, stream>>>(d_out, d_in, n);
    else {
       size_t block_dim = 512;
       size_t num_blocks = ceil(n / (float)block_dim);
       T * scanValues;
       unsigned int * BlockCounter;
       unsigned int * flags;
       size_t flag_size = sizeof(unsigned int) * (num_blocks + 1);
       CUDA_OK(cudaMalloc(&scanValues, sizeof(T) * (num_blocks + 1)));
       CUDA_OK(cudaMemset(scanValues, 0, sizeof(T)));
       CUDA_OK(cudaMalloc(&BlockCounter, sizeof(unsigned int)));
       CUDA_OK(cudaMemset(BlockCounter, 0, sizeof(unsigned int)));
       CUDA_OK(cudaMalloc(&flags, flag_size));
       CUDA_OK(cudaMemset(flags, 0, flag_size));
       CUDA_OK(cudaMemset(flags, 1, sizeof(unsigned int)));
       scan_kernels::Scan<<<num_blocks, block_dim, 0, stream>>>(d_out, d_in, n, scanValues, BlockCounter, flags);
       CUDA_OK(cudaFree(scanValues));
       CUDA_OK(cudaFree(BlockCounter));
       CUDA_OK(cudaFree(flags));
    }
    return CUDA_SUCCESS_MOON;
}


extern "C" RustCudaError scan_baby_bear(bb31_t * d_out, bb31_t* d_in, size_t n, cudaStream_t stream) {
    return ScanTemplate(d_out, d_in, n, stream);
}

extern "C" RustCudaError scan_baby_bear_challenge(bb31_extension_t * d_out, 
    bb31_extension_t  *d_in, size_t n, cudaStream_t stream) {
    return ScanTemplate(d_out, d_in, n, stream);
}