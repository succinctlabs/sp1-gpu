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

__global__ void BaseSumTemplate(const bb31_t* in, bb31_t* out, size_t n, cudaStream_t stream) {
    extern __shared__ bb31_t sdata_base[512];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    if (i < n){
    sdata_base[tid] = in[i];
    }
    __syncthreads();
    
    // Perform block-wise reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_base[tid] = sdata_base[tid] + sdata_base[tid + s];
        }
        __syncthreads();
    }
    
    // Write the result for this block to global memory
    if (tid == 0) {
        out[blockIdx.x] = sdata_base[0];
    }
}

void base_cuda_sum_host_function(bb31_t* a_d, size_t n, bb31_t* out, cudaStream_t stream) {
    bb31_t *result_d;
    size_t new_size = n;

    size_t block_dim = 512;
    size_t num_blocks = ceil(n / (float)block_dim);

    size_t num_rounds = ((size_t)ceil(log2((double)n)))/log2(block_dim);

    for (int i = 0; i < num_rounds; i++) {
        size_t result_size = (new_size + block_dim-1) / block_dim; // Equivalent to div_ceil(new_size, 512)
        
        // Allocate result buffer
        cudaMalloc(&result_d, result_size * sizeof(bb31_t));
        
        // Call the CUDA kernel
        BaseSumTemplate<<<num_blocks, block_dim, 0, stream>>>(a_d, result_d, new_size, stream);

        // Free the old input buffer if it's not the original
        if (i > 0) {
            cudaFree(a_d);
        }
        
        // Update pointers and size for next iteration
        a_d = result_d;
        new_size = result_size;
    }
    // Copy the result back to the host
    cudaMemcpy(out, a_d, new_size*sizeof(bb31_t), cudaMemcpyDeviceToHost);
}

__global__ void ExtensionSumTemplate(const bb31_extension_t* in, bb31_extension_t* out, size_t n, cudaStream_t stream) {
    extern __shared__ bb31_extension_t sdata_extension[512];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    if (i < n){
    sdata_extension[tid] = in[i];
    }
    __syncthreads();
    
    // Perform block-wise reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_extension[tid] = sdata_extension[tid] + sdata_extension[tid + s];
        }
        __syncthreads();
    }
    
    // Write the result for this block to global memory
    if (tid == 0) {
        out[blockIdx.x] = sdata_extension[0];
    }
}

void extension_cuda_sum_host_function(bb31_extension_t* a_d, size_t n, bb31_extension_t* out, cudaStream_t stream) {
    bb31_extension_t *result_d;
    size_t new_size = n;

    size_t block_dim = 512;
    size_t num_blocks = ceil(n / (float)block_dim);

    size_t num_rounds = ((size_t)ceil(log2((double)n)))/log2(block_dim);

    for (int i = 0; i < num_rounds; i++) {
        size_t result_size = (new_size + block_dim-1) / block_dim; // Equivalent to div_ceil(new_size, 512)
        
        // Allocate result buffer
        cudaMalloc(&result_d, result_size * sizeof(bb31_extension_t));
        
        // Call the CUDA kernel
        ExtensionSumTemplate<<<num_blocks, block_dim, 0, stream>>>(a_d, result_d, new_size, stream);

        // Free the old input buffer if it's not the original
        if (i > 0) {
            cudaFree(a_d);
        }
        
        // Update pointers and size for next iteration
        a_d = result_d;
        new_size = result_size;
    }
    // Copy the result back to the host
    cudaMemcpy(out, a_d, new_size*sizeof(bb31_extension_t), cudaMemcpyDeviceToHost);
}

void __global__ dummy_challenge_fn(bb31_extension_t * in, size_t n, bb31_extension_t * result, cudaStream_t stream) {
    extern __shared__ bb31_extension_t sdata[512];
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            sdata[threadIdx.x] = in[i];
            result[i] = sdata[threadIdx.x];
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
    base_cuda_sum_host_function(in, n, result, stream);
}

extern "C" void sum_baby_bear_vec_challenge(bb31_extension_t * in, bb31_extension_t * result, size_t n, cudaStream_t stream) {
    extension_cuda_sum_host_function(in, n, result, stream);
}

