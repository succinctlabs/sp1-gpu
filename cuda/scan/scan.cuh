#pragma once

#include "../fields/bb31_extension_t.cuh"
#include "../utils/exception.cuh"
#include "kernels.cu"

template<typename T>
RustCudaError ScanTemplateSmall(T* d_out, T* d_in, size_t n, cudaStream_t stream) {
    if ((2 * n) <= scan_kernel_small::SECTION_SIZE) {
        scan_kernel_small::SingleBlockScan<<<1, n, 0, stream>>>(d_out, d_in, n);
    } else {
        size_t block_dim = 512;
        size_t num_blocks = ceil(n / (float)block_dim);
        T* scanValues;
        unsigned int* BlockCounter;
        unsigned int* flags;
        size_t flag_size = sizeof(unsigned int) * (num_blocks + 1);
        CUDA_OK(
            cudaMallocAsync(&scanValues, sizeof(T) * (num_blocks + 1), stream)
        );
        CUDA_OK(cudaMemsetAsync(scanValues, 0, sizeof(T), stream));
        CUDA_OK(cudaMallocAsync(&BlockCounter, sizeof(unsigned int), stream));
        CUDA_OK(cudaMemsetAsync(BlockCounter, 0, sizeof(unsigned int), stream));
        CUDA_OK(cudaMallocAsync(&flags, flag_size, stream));
        CUDA_OK(cudaMemsetAsync(flags, 0, flag_size, stream));
        CUDA_OK(cudaMemsetAsync(flags, 1, sizeof(unsigned int), stream));
        scan_kernel_small::Scan<<<num_blocks, block_dim, 0, stream>>>(
            d_out,
            d_in,
            n,
            scanValues,
            BlockCounter,
            flags
        );
        CUDA_OK(cudaFreeAsync(scanValues, stream));
        CUDA_OK(cudaFreeAsync(BlockCounter, stream));
        CUDA_OK(cudaFreeAsync(flags, stream));
    }
    return CUDA_SUCCESS_MOON;
}

template<typename T>
RustCudaError ScanTemplateLarge(T* d_out, T* d_in, size_t n, cudaStream_t stream) {
    if ((2 * n) <= scan_kernel_large::SECTION_SIZE) {
        scan_kernel_large::SingleBlockScan<<<1, n, 0, stream>>>(d_out, d_in, n);
    } else {
        size_t block_dim = 64;
        size_t num_blocks = ceil(n / (float)block_dim);
        T* scanValues;
        unsigned int* BlockCounter;
        unsigned int* flags;
        size_t flag_size = sizeof(unsigned int) * (num_blocks + 1);
        CUDA_OK(
            cudaMallocAsync(&scanValues, sizeof(T) * (num_blocks + 1), stream)
        );
        CUDA_OK(cudaMemsetAsync(scanValues, 0, sizeof(T), stream));
        CUDA_OK(cudaMallocAsync(&BlockCounter, sizeof(unsigned int), stream));
        CUDA_OK(cudaMemsetAsync(BlockCounter, 0, sizeof(unsigned int), stream));
        CUDA_OK(cudaMallocAsync(&flags, flag_size, stream));
        CUDA_OK(cudaMemsetAsync(flags, 0, flag_size, stream));
        CUDA_OK(cudaMemsetAsync(flags, 1, sizeof(unsigned int), stream));
        scan_kernel_large::Scan<<<num_blocks, block_dim, 0, stream>>>(
            d_out,
            d_in,
            n,
            scanValues,
            BlockCounter,
            flags
        );
        CUDA_OK(cudaFreeAsync(scanValues, stream));
        CUDA_OK(cudaFreeAsync(BlockCounter, stream));
        CUDA_OK(cudaFreeAsync(flags, stream));
    }
    return CUDA_SUCCESS_MOON;
}

extern "C" RustCudaError
scan_baby_bear(bb31_t* d_out, bb31_t* d_in, size_t n, cudaStream_t stream) {
    return ScanTemplateSmall(d_out, d_in, n, stream);
}

extern "C" RustCudaError scan_baby_bear_challenge(
    bb31_extension_t* d_out,
    bb31_extension_t* d_in,
    size_t n,
    cudaStream_t stream
) {
    return ScanTemplateSmall(d_out, d_in, n, stream);
}