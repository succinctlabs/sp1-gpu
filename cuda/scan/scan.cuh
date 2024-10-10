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
       CUDA_OK(cudaMallocAsync(&scanValues, sizeof(T) * (num_blocks + 1), stream));
       CUDA_OK(cudaMemsetAsync(scanValues, 0, sizeof(T), stream));
       CUDA_OK(cudaMallocAsync(&BlockCounter, sizeof(unsigned int), stream));
       CUDA_OK(cudaMemsetAsync(BlockCounter, 0, sizeof(unsigned int), stream));
       CUDA_OK(cudaMallocAsync(&flags, flag_size, stream));
       CUDA_OK(cudaMemsetAsync(flags, 0, flag_size, stream));
       CUDA_OK(cudaMemsetAsync(flags, 1, sizeof(unsigned int), stream));
       scan_kernels::Scan<<<num_blocks, block_dim, 0, stream>>>(d_out, d_in, n, scanValues, BlockCounter, flags);
       CUDA_OK(cudaFreeAsync(scanValues, stream));
       CUDA_OK(cudaFreeAsync(BlockCounter, stream));
       CUDA_OK(cudaFreeAsync(flags, stream));
    }
    return CUDA_SUCCESS_MOON;
}

template<typename T> __global__ void AddTemplate(T * a, T * b, T * c, size_t n, cudaStream_t stream){
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n) {
        c[index] = a[index] + b[index];
    } 
}



template<typename T>
__global__ void SumTemplate(const T* in, T* out, size_t n, cudaStream_t stream) {
    extern __shared__ T sdata[512];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    if (i < n){
    sdata[tid] = in[i];
    }
    __syncthreads();
    
    // Perform block-wise reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write the result for this block to global memory
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

template<typename T>
__device__ T compute_eq_poly(size_t i, T * point, size_t n_variables) {
    T result = T::one();
    for(size_t j = 0; j < n_variables; j++) {
        bool selector = (i>>j & 1)==1;
        result *= T(selector) * point[j] + T(!selector) * (T::one() - point[j]); 
    }
    return result;
}

template<typename T>
__global__ void ComputeEqPolyTemplate(T * d_out, T * point, size_t n_variables) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < (1<<n_variables)) {
        d_out[index] = compute_eq_poly(index, point, n_variables);
    }
}

extern "C" RustCudaError scan_baby_bear(bb31_t * d_out, bb31_t* d_in, size_t n, cudaStream_t stream) {
    return ScanTemplate(d_out, d_in, n, stream);
}

extern "C" RustCudaError scan_baby_bear_challenge(bb31_extension_t * d_out, 
    bb31_extension_t  *d_in, size_t n, cudaStream_t stream) {
    return ScanTemplate(d_out, d_in, n, stream);
}

extern "C" void add_baby_bear_vecs(bb31_t * a, bb31_t * b, bb31_t * c, size_t n, cudaStream_t stream) {
    size_t block_dim = 512;
    size_t num_blocks = ceil(n / (float)block_dim);
    AddTemplate<<<num_blocks, block_dim, 0, stream>>>(a, b, c, n, stream);
}

extern "C" void sum_baby_bear_vec(bb31_t * in, bb31_t * result, size_t n, cudaStream_t stream) {
    size_t block_dim = 512;
    size_t num_blocks = ceil(n/ (float)block_dim);
    SumTemplate<<<num_blocks, block_dim, 0, stream>>>(in, result, n, stream);
}

extern "C" void compute_eq_poly(bb31_t * d_out, bb31_t * point, size_t n_variables, cudaStream_t stream){
    size_t block_dim = 512;
    size_t num_blocks = ceil((1<<n_variables)/ (float)block_dim);
    ComputeEqPolyTemplate<<<num_blocks, block_dim, 0, stream>>>(d_out, point, n_variables);
}
