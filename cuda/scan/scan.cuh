#pragma once

#include <bit>

#include "../fields/bb31_extension_t.cuh"
#include "../utils/exception.cuh"
#include "kernels.cu"
#include "moongate_cuda_cbindgen.hpp"

template<typename T>
RustCudaError
ScanTemplate(T* d_out, const T* d_in, size_t n, cudaStream_t stream) {
    if ((2 * n) <= scan_kernels::SECTION_SIZE)
        scan_kernels::SingleBlockScan<<<1, n, 0, stream>>>(d_out, d_in, n);
    else {
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
        scan_kernels::Scan<<<num_blocks, block_dim, 0, stream>>>(
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

namespace moongate {

CudaRustError
scan_baby_bear(F* a, const F* b, uintptr_t n, CudaStreamHandle stream) {
    return ScanTemplate(
        std::bit_cast<bb31_t*>(a),
        std::bit_cast<const bb31_t*>(b),
        n,
        std::bit_cast<cudaStream_t>(stream)
    );
}

CudaRustError scan_baby_bear_challenge(
    EF* a,
    const EF* b,
    uintptr_t n,
    CudaStreamHandle stream
) {
    return ScanTemplate(
        std::bit_cast<bb31_extension_t*>(&a->value),
        std::bit_cast<const bb31_extension_t*>(b),
        n,
        std::bit_cast<cudaStream_t>(stream)
    );
}

}  // namespace moongate
