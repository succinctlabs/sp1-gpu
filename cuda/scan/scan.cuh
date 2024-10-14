#pragma once

#include "../fields/bb31_extension_t.cuh"
#include "../utils/exception.cuh"
#include "kernels.cu"

template<typename T>
RustCudaError ScanTemplate(T* d_out, T* d_in, size_t n, cudaStream_t stream) {
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

template<typename T>
__global__ void AddTemplate(T* a, T* b, T* c, size_t n, cudaStream_t stream) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

template<typename T>
__device__ __forceinline__ T
ComputeEqPolyVal(size_t i, T* point, size_t n_variables) {
    T result = T::one();
    for (size_t j = 0; j < n_variables; j++) {
        bool selector = (i >> j & 1) == 1;
        result *= T(selector) * point[n_variables - 1 - j]
            + T(!selector) * (T::one() - point[n_variables - 1 - j]);
    }
    return result;
}

template<typename T, typename S, typename R>
__global__ void
HadamardProduct(const T* a, const S* b, R* c, size_t n_low, size_t n_high) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 1 << n_low) {
        for (size_t i = 0; i < 1 << n_high; i++) {
            c[index + i * (1 << n_low)] =
                b[index + i * (1 << n_low)] * a[index + i * (1 << n_low)];
        }
    }
}

template<typename T>
__global__ void
ComputeEqPolyTemplate(T* d_out, T* point, size_t n_low, size_t n_high) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 1 << n_low) {
        T base_val = ComputeEqPolyVal(index, point + n_high, n_low);
        for (size_t i = 0; i < 1 << n_high; i++) {
            d_out[index + i * (1 << n_low)] =
                base_val * ComputeEqPolyVal(i, point, n_high);
        }
    }
}

// Compute the eq polynomial and take the Hadamard product of the result with the input in place.
// Generic in the type T.
template<typename T>
__global__ void
ComputeEqHadamard(T* point, T* in, size_t n_low, size_t n_high) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 1 << n_low) {
        T base_val = ComputeEqPolyVal(index, point + n_high, n_low);
        for (size_t i = 0; i < 1 << n_high; i++) {
            in[index + i * (1 << n_low)] = base_val
                * ComputeEqPolyVal(i, point, n_high)
                * in[index + i * (1 << n_low)];
        }
    }
}

__global__ void
BaseSumKernel(const bb31_t* in, bb31_t* out, size_t n) {
    extern __shared__ bb31_t sdata_base[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    if (i < n) {
        sdata_base[tid] = in[i];
    } else {
        sdata_base[tid] = bb31_t(0);  // Zero initialization for padding
    }
    __syncthreads();

    // Perform block-wise reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            sdata_base[tid] = sdata_base[tid] + sdata_base[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        out[blockIdx.x] = sdata_base[0];
    }
}

void base_cuda_sum_host_function(
    bb31_t* a_d,
    size_t n,
    bb31_t* out,
    cudaStream_t stream
) {
    size_t new_size = n;
    bb31_t *result_d, *temp_d;
    const size_t block_dim = 512;

    size_t num_rounds =
        (size_t)ceil(log2((double)n + 1) / log2((double)block_dim));

    // Allocate temporary buffer
    cudaMalloc(&temp_d, n * sizeof(bb31_t));

    for (size_t i = 0; i < num_rounds - 1; i++) {
        size_t num_blocks = (new_size + block_dim - 1) / block_dim;
        size_t result_size = num_blocks;

        // Allocate result buffer
        cudaMalloc(&result_d, result_size * sizeof(bb31_t));

        // Call the CUDA kernel
        BaseSumKernel<<<
            num_blocks,
            block_dim,
            block_dim * sizeof(bb31_t),
            stream>>>((i == 0) ? a_d : temp_d, result_d, new_size);

        cudaStreamSynchronize(stream);

        // Swap pointers for next iteration
        cudaFree(temp_d);
        temp_d = result_d;

        new_size = result_size;
    }

    // Copy the result back to the host
    cudaMemcpyAsync(
        out,
        temp_d,
        new_size * sizeof(bb31_t),
        cudaMemcpyDeviceToHost,
        stream
    );
    cudaStreamSynchronize(stream);

    // Clean up
    cudaFree(temp_d);
}

// Kernel function
__global__ void ExtensionSumKernel(
    const bb31_extension_t* in,
    bb31_extension_t* out,
    size_t n
) {
    extern __shared__ bb31_extension_t sdata_extension[];
    unsigned int tid = threadIdx.x;

    // Load input into shared memory, performing 1<<n_high loads per thread.
    unsigned int segment = blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    if (i < n) {
        sdata_extension[tid] = in[i];
    } else {
        sdata_extension[tid] =
            bb31_extension_t(0);  // Zero initialization for padding
    }

    __syncthreads();

    // Perform block-wise reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < n) {
            sdata_extension[tid] =
                sdata_extension[tid] + sdata_extension[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        out[blockIdx.x] = sdata_extension[0];
    }
}

// Host function
void extension_cuda_sum_host_function(
    bb31_extension_t* a_d,
    size_t n,
    bb31_extension_t* out,
    cudaStream_t stream
) {
    size_t new_size = n;
    bb31_extension_t *result_d, *temp_d;
    const size_t block_dim = 512;

    size_t num_rounds =
        (size_t)ceil(log2((double)n + 1) / log2((double)block_dim));

    // Allocate temporary buffer
    cudaMalloc(&temp_d, n * sizeof(bb31_extension_t));

    for (size_t i = 0; i < num_rounds - 1; i++) {
        size_t num_blocks = (new_size + block_dim - 1) / block_dim;
        size_t result_size = num_blocks;

        // Allocate result buffer
        cudaMalloc(&result_d, result_size * sizeof(bb31_extension_t));

        // Call the CUDA kernel
        ExtensionSumKernel<<<
            num_blocks,
            block_dim,
            block_dim * sizeof(bb31_extension_t),
            stream>>>((i == 0) ? a_d : temp_d, result_d, new_size);

        cudaStreamSynchronize(stream);

        // Swap pointers for next iteration
        cudaFree(temp_d);
        temp_d = result_d;

        new_size = result_size;
    }

    // Copy the result back to the host
    cudaMemcpyAsync(
        out,
        temp_d,
        new_size * sizeof(bb31_extension_t),
        cudaMemcpyDeviceToHost,
        stream
    );
    cudaStreamSynchronize(stream);

    // Clean up
    cudaFree(temp_d);
}

template<typename T>
void extension_multilinear_eval(
    bb31_extension_t* out,
    bb31_extension_t* point,
    T* in,
    size_t n_low,
    size_t n_high,
    cudaStream_t stream
) {
    size_t block_dim = 512;
    size_t num_blocks = ceil((1 << (n_low)) / (float)block_dim);
    size_t n = 1 << (n_low + n_high);

    ComputeEqHadamard<<<num_blocks, block_dim, 0, stream>>>(
        point,
        in,
        n_low,
        n_high
    );

    num_blocks = ceil(n / (float)block_dim);

    cudaStreamSynchronize(stream);

    extension_cuda_sum_host_function(in, n, out, stream);
}

extern "C" RustCudaError
scan_baby_bear(bb31_t* d_out, bb31_t* d_in, size_t n, cudaStream_t stream) {
    return ScanTemplate(d_out, d_in, n, stream);
}

extern "C" RustCudaError scan_baby_bear_challenge(
    bb31_extension_t* d_out,
    bb31_extension_t* d_in,
    size_t n,
    cudaStream_t stream
) {
    return ScanTemplate(d_out, d_in, n, stream);
}

extern "C" void add_baby_bear_vecs(
    bb31_t* a,
    bb31_t* b,
    bb31_t* c,
    size_t n,
    cudaStream_t stream
) {
    size_t block_dim = 512;
    size_t num_blocks = ceil(n / (float)block_dim);
    AddTemplate<<<num_blocks, block_dim, 0, stream>>>(a, b, c, n, stream);
}

extern "C" void compute_eq_poly(
    bb31_t* d_out,
    bb31_t* point,
    size_t n_low,
    size_t n_high,
    cudaStream_t stream
) {
    size_t block_dim = 512;
    size_t num_blocks = ceil((1 << n_low) / (float)block_dim);
    ComputeEqPolyTemplate<<<num_blocks, block_dim, 0, stream>>>(
        d_out,
        point,
        n_low,
        n_high
    );
}

extern "C" void compute_extension_eq_poly(
    bb31_extension_t* d_out,
    bb31_extension_t* point,
    size_t n_low,
    size_t n_high,
    cudaStream_t stream
) {
    size_t block_dim = 512;
    size_t num_blocks = ceil((1 << n_low) / (float)block_dim);
    ComputeEqPolyTemplate<<<num_blocks, block_dim, 0, stream>>>(
        d_out,
        point,
        n_low,
        n_high
    );
}

extern "C" void hadamard_product(
    bb31_t* a,
    bb31_t* b,
    bb31_t* c,
    size_t n_low,
    size_t n_high,
    cudaStream_t stream
) {
    size_t block_dim = 512;
    size_t num_blocks = ceil((1 << (n_low + n_high)) / (float)block_dim);
    HadamardProduct<<<num_blocks, block_dim, 0, stream>>>(
        a,
        b,
        c,
        n_low,
        n_high
    );
}

extern "C" void ef_hadamard_product(
    bb31_t* a,
    bb31_extension_t* b,
    bb31_extension_t* c,
    size_t n_low,
    size_t n_high,
    cudaStream_t stream
) {
    size_t block_dim = 512;
    size_t num_blocks = ceil((1 << (n_low + n_high)) / (float)block_dim);
    HadamardProduct<<<num_blocks, block_dim, 0, stream>>>(
        a,
        b,
        c,
        n_low,
        n_high
    );
}

extern "C" void
sum_baby_bear_vec(bb31_t* in, bb31_t* result, size_t n, cudaStream_t stream) {
    base_cuda_sum_host_function(in, n, result, stream);
}

extern "C" void sum_baby_bear_vec_challenge(
    bb31_extension_t* in,
    bb31_extension_t* result,
    size_t n,
    cudaStream_t stream
) {
    extension_cuda_sum_host_function(in, n, result, stream);
}

extern "C" void extension_multilinear_evaluator(
    bb31_extension_t* out,
    bb31_extension_t* point,
    bb31_extension_t* in,
    size_t n_low,
    size_t n_high,
    cudaStream_t stream
) {
    extension_multilinear_eval(out, point, in, n_low, n_high, stream);
}
