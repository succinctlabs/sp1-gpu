#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
#include "../fields/bb31_t.cuh"
#include "../fields/bb31_extension_t.cuh"
#include "../utils/exception.cuh"

namespace cg = cooperative_groups;


template<typename F> __device__ int reduce_sum(cg::thread_group g, F *temp, F val)
{
    int lane = g.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync();
        if(lane<i) val += temp[lane + i];
        g.sync(); 
    }
    return val; // note: only thread 0 will return full sum
}

template<typename F> __global__ void partialBlockSum(F* A, F* partial_sums, size_t len) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    F thread_sum = F::zero();

    // Stride loop to accumulate partial sum
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        thread_sum += A[i];
    }

    // Warp-level reduction within tiles
    thread_sum = cg::reduce(tile, thread_sum, cg::plus<F>());

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    F* shared_sum = reinterpret_cast<F*>(memory);

    // Only the first thread of each warp writes to shared memory
    if (tile.thread_rank() == 0) {
        shared_sum[tile.meta_group_rank()] = thread_sum;
    }
    block.sync();  // Synchronize after warp-level reduction

    // Perform tree-based reduction on shared memory
    for (int stride = (block.size() / tile.size()) / 2; stride > 0; stride /= 2) {
        if (block.thread_rank() < stride) {
            shared_sum[block.thread_rank()] += shared_sum[block.thread_rank() + stride];
        }
        block.sync();  // Synchronize after each step
    }

    // Write the result to the partial_sums array
    if (block.thread_rank() == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}

template<typename F, typename EF> __global__ void partialBlockSumExtension(
    EF* A, 
    EF* partial_sums, 
    size_t len) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    EF thread_sum = EF::zero();

    // Stride loop to accumulate partial sum
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        thread_sum += A[i];
    }

    // Warp-level reduction within tiles
    // thread_sum = cg::reduce(tile, thread_sum, cg::plus<F>());
    #pragma unroll
    for(int j=0 ; j<= EF::D; j++) {
        cuda::atomic_ref<F, cuda::thread_scope_block> atomic(thread_sum.value[j]);
        cg::reduce_store_async(tile, atomic, thread_sum.value[j], cg::plus<F>());
    }
    tile.sync();

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    EF* shared_sum = reinterpret_cast<EF*>(memory);

    // Only the first thread of each warp writes to shared memory
    if (tile.thread_rank() == 0) {
        shared_sum[tile.meta_group_rank()] = thread_sum;
    }
    block.sync();  // Synchronize after warp-level reduction

    // Perform tree-based reduction on shared memory
    for (int stride = (block.size() / tile.size()) / 2; stride > 0; stride /= 2) {
        if (block.thread_rank() < stride) {
            shared_sum[block.thread_rank()] += shared_sum[block.thread_rank() + stride];
        }
        block.sync();  // Synchronize after each step
    }

    // Write the result to the partial_sums array
    if (block.thread_rank() == 0) {
        partial_sums[blockIdx.x] = shared_sum[0];
    }
}


template<typename F, int N> __global__ void partialMatrixBlockSum(
    F* A, 
    F* partial_sums,
    size_t width, 
    size_t height) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    size_t start_col = N * (blockDim.y * blockIdx.y + threadIdx.y);
    if (start_col >= width) return;

    F thread_sums[N];

    #pragma unroll
    for (int j =0; j < N; j++) { 
        thread_sums[j] = F::zero();
    }

    // Stride loop to accumulate partial sum
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height; i += blockDim.x * gridDim.x) {
        #pragma unroll
        for (int j =0; j < N; j++) {
            int col = start_col + j;
            if (col < width) { 
               thread_sums[j] += A[col * height + i];
            }
        }
    }

    // Warp-level reduction for each column
    #pragma unroll
    for (int j =0; j < N; j++) { 
        thread_sums[j] = cg::reduce(tile, thread_sums[j], cg::plus<F>());
        cuda::atomic_ref<F, cuda::thread_scope_block> atomic(thread_sums[j]);
        cg::reduce_update_async(tile, atomic, thread_sums[j], cg::plus<F>());
    }
    tile.sync();

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    F* shared_sums = reinterpret_cast<F*>(memory);

    // Only the first thread of each warp writes to shared memory
    if (tile.thread_rank() == 0) {
        #pragma unroll
        for (int j =0; j < N; j++) { 
          shared_sums[j + N * tile.meta_group_rank()] = thread_sums[j];
        }
    }
    block.sync();  // Synchronize after warp-level reduction

    // Perform tree-based reduction on shared memory
    for (int stride = (block.size() / tile.size()) / 2; stride > 0; stride /= 2) {
        if (block.thread_rank() < stride) {
            #pragma unroll
            for (int j =0; j < N; j++) { 
              shared_sums[j + N * block.thread_rank()] += shared_sums[j + N * (block.thread_rank() + stride)];
            }
        }
        block.sync();  // Synchronize after each step
    }

    // Write the result to the partial_sums array
    if (block.thread_rank() == 0) {
        #pragma unroll
        for (int j =0; j < N; j++) { 
            int col = start_col + j;
            if (col < width) { 
                partial_sums[(start_col + j) * blockDim.x + blockIdx.x] = shared_sums[j];
            }
        }
    }
}

template<typename F, int N> __global__ void blockSumMatrix(
    F* A,
   F* total_sums, 
   size_t width,
   size_t height) {

    // Starting column index for this block
    int start_col = N * (blockIdx.y * blockDim.y + threadIdx.y);
    if (start_col >= width) return;

    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    F thread_sums[N];

    #pragma unroll
    for (int j =0; j < N; j++) { 
        thread_sums[j] = F::zero();
    }
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height; i += blockDim.x * gridDim.x) {
        #pragma unroll
        for (int j = 0; j < N; j++) {
            int col = start_col + j;
            if (col < width) {
                thread_sums[j] += A[col * height + i];
            }
        }
    }
    #pragma unroll
    for (int j =0; j < N; j++) { 
      // reduce thread sums across the tile, add the result to the atomic
      cuda::atomic_ref<F, cuda::thread_scope_block> atomic(total_sums[j]);
      cg::reduce_update_async(tile, atomic, thread_sums[start_col + j], cg::plus<F>());
    }
    // synchronize the block, to ensure all async reductions are ready
    block.sync();
}


template<typename F> __global__ void blockSum(F* A, F* total_sum, size_t len) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    F thread_sum = F::zero();
    
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        thread_sum += A[i];
    }
    cuda::atomic_ref<F, cuda::thread_scope_block> atomic(total_sum[0]);
    // reduce thread sums across the tile, add the result to the atomic
    cg::reduce_update_async(tile, atomic, thread_sum, cg::plus<F>());
    // synchronize the block, to ensure all async reductions are ready
    block.sync();
}

template<typename F, typename EF> __global__ void blockSumExtension(EF* A, EF* total_sum, size_t len) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    EF thread_sum = EF::zero();
    
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        thread_sum += A[i];
    }

    #pragma unroll
    for(int j=0 ; j<= EF::D; j++) {
        cuda::atomic_ref<F, cuda::thread_scope_block> atomic(total_sum[0].value[j]);
        cg::reduce_update_async(tile, atomic, thread_sum.value[j], cg::plus<F>());
    }
    block.sync();
}

template<typename F> RustCudaError vectorSum(F* in, F* result, size_t len, cudaStream_t stream) {
    size_t numThreads = 512;
    size_t numBlocks = (((len - 1)/numThreads + 1) - 1) / 8 + 1;

    // Allocate the partial sums and set them to zero. 
    F * partial_sums;

    CUDA_OK(cudaMallocAsync(&partial_sums, sizeof(F) * numBlocks, stream));

    size_t numTiles = numThreads/32;
    partialBlockSum<<<numBlocks, numThreads, numTiles * sizeof(F), stream>>>(in, partial_sums, len);
    
    size_t new_len = numBlocks;
    numBlocks = (((new_len - 1)/numThreads + 1) - 1) / 8 + 1;
    blockSum<<<numBlocks, numThreads, 0, stream>>>(partial_sums, result, new_len);

    CUDA_OK(cudaFreeAsync(partial_sums, stream));

    return CUDA_SUCCESS_MOON;
}

template<typename F, typename EF> RustCudaError vectorSumExtension(EF* in, EF* result, size_t len, cudaStream_t stream) {
    size_t numThreads = 512;
    size_t numBlocks = (((len - 1)/numThreads + 1) - 1) / 8 + 1;

    // Allocate the partial sums and set them to zero. 
    EF * partial_sums;

    CUDA_OK(cudaMallocAsync(&partial_sums, sizeof(EF) * numBlocks, stream));

    size_t numTiles = numThreads/32;
    partialBlockSumExtension<F, EF><<<numBlocks, numThreads, numTiles * sizeof(EF), stream>>>(in, partial_sums, len);
    
    size_t new_len = numBlocks;
    numBlocks = (((new_len - 1)/numThreads + 1) - 1) / 8 + 1;
    blockSumExtension<F, EF><<<numBlocks, numThreads, 0, stream>>>(partial_sums, result, new_len);

    CUDA_OK(cudaFreeAsync(partial_sums, stream));

    return CUDA_SUCCESS_MOON;
}

extern "C" RustCudaError vectorSumBabyBear(bb31_t* in, bb31_t* result, size_t len, cudaStream_t stream) {
    return vectorSum(in, result, len, stream);
}

extern "C" RustCudaError vectorSumBabyBearExtension(bb31_extension_t* in, bb31_extension_t* result, size_t len, cudaStream_t stream) {
    return vectorSumExtension<bb31_t, bb31_extension_t>(in, result, len, stream);
}

