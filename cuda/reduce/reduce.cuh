#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
#include "../fields/bb31_t.cuh"
#include "../fields/bb31_extension_t.cuh"
#include "../utils/exception.cuh"

#include <stdio.h>

namespace cg = cooperative_groups;


template <typename Ty>
 struct AddOp {
    __device__ __forceinline__ Ty operator()(const Ty arg1, const Ty arg2) const {
        return arg1 + arg2;
    }

    __device__ __forceinline__ void operator()(Ty& arg1, const Ty arg2) const {
        arg1 += arg2;
    }

    template<typename TyGroup>
    __device__ __forceinline__ Ty reduce(const TyGroup& group, Ty&& val) {
        return cg::reduce(group, val, cg::plus<Ty>());
    }
 };


template <>
  struct AddOp<bb31_t> {
    template<typename TyGroup>
    __device__ __forceinline__ void final_block_reduction_async(const TyGroup& group, bb31_t* dst, bb31_t val) {
        cuda::atomic_ref<bb31_t, cuda::thread_scope_block> atomic(dst[0]);
        // reduce thread sums across the tile, add the result to the atomic
        return cg::reduce_update_async(group, atomic, val, cg::plus<bb31_t>());
    }
  };


template <>
  struct AddOp<bb31_extension_t> {
    template<typename TyGroup>
    __device__ __forceinline__ void final_block_reduction_async(const TyGroup& group, bb31_extension_t* dst, bb31_extension_t val) {
        // Split the extension into a slice of base field elements and make a separate atomic update.
        #pragma unroll
        for(int j = 0 ; j<= bb31_extension_t::D; j++) {
            cuda::atomic_ref<bb31_t, cuda::thread_scope_block> atomic(dst[0].value[j]);
            cg::reduce_update_async(group, atomic, val.value[j], cg::plus<bb31_t>());
        }
    }
};


template<typename F> __global__ void partialBlockReduce(F* A, F* partial_sums, size_t len) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    F thread_sum = F::zero();

    // Stride loop to accumulate partial sum
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        thread_sum += A[i];
    }

    // Warp-level reduction within tiles
    thread_sum = cg::reduce(tile, thread_sum, cg::plus<F>());
    // AddOp<bb31_t> op;
    // thread_sum = op.reduce(tile, thread_sum);

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


template<typename F> __global__ void blockReduce(F* A, F* total_sum, size_t len) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    F thread_sum = F::zero();
    
    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        thread_sum += A[i];
    }

    AddOp<F> op;   
    op.final_block_reduction_async(tile, total_sum, thread_sum); 
    block.sync();
}


template<typename F> RustCudaError vectorSum(F* in, F* result, size_t len, cudaStream_t stream) {
    size_t numThreads = 512;
    size_t numBlocks = (((len - 1)/numThreads + 1) - 1) / 8 + 1;

    // Allocate the partial sums and set them to zero. 
    F * partial_sums;

    CUDA_OK(cudaMallocAsync(&partial_sums, sizeof(F) * numBlocks, stream));

    size_t numTiles = numThreads/32;
    partialBlockReduce<<<numBlocks, numThreads, numTiles * sizeof(F), stream>>>(in, partial_sums, len);
    
    size_t new_len = numBlocks;
    numBlocks = (((new_len - 1)/numThreads + 1) - 1) / 8 + 1;
    blockReduce<<<numBlocks, numThreads, 0, stream>>>(partial_sums, result, new_len);

    CUDA_OK(cudaFreeAsync(partial_sums, stream));

    return CUDA_SUCCESS_MOON;
}


extern "C" RustCudaError vectorSumBabyBear(bb31_t* in, bb31_t* result, size_t len, cudaStream_t stream) {
    return vectorSum(in, result, len, stream);
}

extern "C" RustCudaError vectorSumBabyBearExtension(bb31_extension_t* in, bb31_extension_t* result, size_t len, cudaStream_t stream) {
    return vectorSum<bb31_extension_t>(in, result, len, stream);
}

