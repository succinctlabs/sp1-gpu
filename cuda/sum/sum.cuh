#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
#include "../fields/bb31_t.cuh"
#include "../fields/bb31_extension_t.cuh"

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

    // Block-wide reduction in shared memory
    if (block.thread_rank() < (block.size() / tile.size())) {
        thread_sum = shared_sum[block.thread_rank()];
    }
    block.sync();  // Synchronize before the final reduction

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

/// @brief An operator for element-wise addition
/// @tparam F 
/// @tparam N 
template <typename F, int N>
struct BatchedSumOp {
    __device__ void operator()(F (&a)[N], const F (&b)[N]) const {
        #pragma unroll
        for (int i = 0; i < N; i++) {
        }
    }
};

template<typename F, int N> __global__ void partialBlockSumBatch(F* A, F* partial_sums, size_t len) {
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

    // Block-wide reduction in shared memory
    if (block.thread_rank() < (block.size() / tile.size())) {
        thread_sum = shared_sum[block.thread_rank()];
    }
    block.sync();  // Synchronize before the final reduction

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

template<typename F> void vectorSum(F* in, F* result, size_t len, cudaStream_t stream) {
    size_t numThreads = 512;
    size_t numBlocks = (((len - 1)/numThreads + 1) - 1) / 8 + 1;

    // Allocate the partial sums and set them to zero. 
    F * partial_sums;

    CUDA_UNWRAP(cudaMallocAsync(&partial_sums, sizeof(F) * numBlocks, stream));

    size_t numTiles = numThreads/32;
    partialBlockSum<<<numBlocks, numThreads, numTiles * sizeof(F), stream>>>(in, partial_sums, len);
    
    size_t new_len = numBlocks;
    numBlocks = (((new_len - 1)/numThreads + 1) - 1) / 8 + 1;
    blockSum<<<numBlocks, numThreads, 0, stream>>>(partial_sums, result, new_len);

    CUDA_UNWRAP(cudaFreeAsync(partial_sums, stream));
}

template<typename F, typename EF> void vectorSumExtension(EF* in, EF* result, size_t len, cudaStream_t stream) {
    size_t numThreads = 512;
    size_t numBlocks = (((len - 1)/numThreads + 1) - 1) / 8 + 1;

    // Allocate the partial sums and set them to zero. 
    EF * partial_sums;

    CUDA_UNWRAP(cudaMallocAsync(&partial_sums, sizeof(EF) * numBlocks, stream));

    size_t numTiles = numThreads/32;
    partialBlockSum<<<numBlocks, numThreads, numTiles * sizeof(EF), stream>>>(in, partial_sums, len);
    
    size_t new_len = numBlocks;
    numBlocks = (((new_len - 1)/numThreads + 1) - 1) / 8 + 1;
    blockSumExtension<F, EF><<<numBlocks, numThreads, 0, stream>>>(partial_sums, result, new_len);

    CUDA_UNWRAP(cudaFreeAsync(partial_sums, stream));
}

extern "C" void vectorSumBabyBear(bb31_t* in, bb31_t* result, size_t len, cudaStream_t stream) {
    vectorSum(in, result, len, stream);
}

extern "C" void vectorSumBabyBearExtension(bb31_extension_t* in, bb31_extension_t* result, size_t len, cudaStream_t stream) {
    vectorSumExtension<bb31_t, bb31_extension_t>(in, result, len, stream);
}

extern "C" void partialSumBabyBear(bb31_t* in, bb31_t*  out, size_t len, cudaStream_t stream) {
    size_t numThreads = 512;
    size_t numBlocks = (((len - 1)/numThreads + 1) - 1) + 1;
    // printf("{}", )
    partialBlockSum<<<numBlocks, numThreads, numThreads * sizeof(bb31_t), stream>>>(in, out, len);
} 

