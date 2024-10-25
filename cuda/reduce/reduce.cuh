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
  struct AddOpFinalReduce {
    template<typename TyGroup>
    __device__ __forceinline__ static void final_block_reduction_async(const TyGroup& group, Ty* dst, Ty val);
};

template <typename Ty>
 struct AddOp {
    __device__ __forceinline__ Ty initial() const {
        return Ty::zero();
    }

    __device__ __forceinline__ Ty operator()(const Ty arg1, const Ty arg2) const {
        return arg1 + arg2;
    }

    __device__ __forceinline__ void evalAssign(Ty& arg1, const Ty arg2) const {
        arg1 += arg2;
    }

    template<typename TyGroup>
    __device__ __forceinline__ Ty reduce(const TyGroup& group, Ty val) {
        return cg::reduce(group, val, cg::plus<Ty>());
    }

    template<typename TyGroup>
    __device__ __forceinline__ void final_block_reduction_async(const TyGroup& group, Ty* dst, Ty val) {
       return AddOpFinalReduce<Ty>::final_block_reduction_async(group, dst, val); 
    }
 };



template <>
  struct AddOpFinalReduce<bb31_t> {
    template<typename TyGroup>
    __device__ __forceinline__ static void final_block_reduction_async(
        const TyGroup& group, 
        bb31_t* dst, 
        bb31_t val) {
        cuda::atomic_ref<bb31_t, cuda::thread_scope_block> atomic(dst[0]);
        // reduce thread sums across the tile, add the result to the atomic
        return cg::reduce_update_async(group, atomic, val, cg::plus<bb31_t>());
    }
  };


template <>
  struct AddOpFinalReduce<bb31_extension_t> {
    template<typename TyGroup>
    __device__ __forceinline__ static void final_block_reduction_async(
        const TyGroup& group, 
        bb31_extension_t* dst, 
        bb31_extension_t val) {
        // Split the extension into a slice of base field elements and make a separate atomic update.
        #pragma unroll
        for(int j = 0 ; j < bb31_extension_t::D; j++) {
            cuda::atomic_ref<bb31_t, cuda::thread_scope_block> atomic(dst[0].value[j]);
            cg::reduce_update_async(group, atomic, val.value[j], cg::plus<bb31_t>());
        }
    }
};

template<typename F, typename TyOp, typename TyBlock, typename TyTile> __device__ F partialBlockReduce(
    const TyBlock& block,
    const TyTile& tile,
    F val,
    F* shared,
    TyOp&& op
) {
    // Warp-level reduction within tiles
    val = op.reduce(tile, val);

    // Only the first thread of each warp writes to shared memory
    if (tile.thread_rank() == 0) {
        shared[tile.meta_group_rank()] = val;
    }
    block.sync();  // Synchronize after warp-level reduction

    // Perform tree-based reduction on shared memory
    for (int stride = (block.size() / tile.size()) / 2; stride > 0; stride /= 2) {
        if (block.thread_rank() < stride) {
            op.evalAssign(shared[block.thread_rank()], shared[block.thread_rank() + stride]);
        }
        block.sync();  // Synchronize after each step
    }

    return shared[0];
}

template<typename F, typename TyOp> __global__ void partialBlockReduceKernel(
    F* A, 
    F* partial, 
    size_t width,
    size_t height,
    TyOp&& op) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    F thread_val = op.initial();

    size_t batchIdx = blockDim.y * blockIdx.y + threadIdx.y;
    if (batchIdx >= width) {
        return;
    }

    // Stride loop to accumulate partial sum
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height; i += blockDim.x * gridDim.x) {
        op.evalAssign(thread_val, A[batchIdx * height + i]);
    }

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    F* shared = reinterpret_cast<F*>(memory);

    // // Warp-level reduction within tiles
    thread_val = partialBlockReduce(block, tile, thread_val, shared, op);

    // Write the result to the partial_sums array
    if (block.thread_rank() == 0) {
        partial[batchIdx * gridDim.x + blockIdx.x] = shared[0];
    }
}


template<typename F, typename TyOp> __global__ void blockReduce(
    F* A, 
    F* result, 
    size_t width,
    size_t height,
    TyOp&& op) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    size_t batchIdx = blockDim.y * blockIdx.y + threadIdx.y;
    if (batchIdx >= width) {
        return;
    }

    F thread_val = op.initial();

    for(size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height; i += blockDim.x * gridDim.x) {
        op.evalAssign(thread_val, A[batchIdx * height + i]);
    }

    op.final_block_reduction_async(tile, &result[batchIdx], thread_val); 
    block.sync();
}


template<typename F> RustCudaError vectorSum(
    F* in, 
    F* result,
    size_t width, 
    size_t height,
    cudaStream_t stream) {
    dim3 blockDim(512, 1, 1);
    size_t numReduceBlocks = (((height - 1)/blockDim.x + 1) - 1) / 8 + 1;
    dim3 gridDim(numReduceBlocks, width, 1);

    // Allocate the partial sums and set them to zero. 
    F * partial_sums;
    CUDA_OK(cudaMallocAsync(&partial_sums, sizeof(F) * gridDim.x * width, stream));

    size_t numTiles = blockDim.x / 32;

    AddOp<F> op;

    partialBlockReduceKernel<<<gridDim, blockDim, numTiles * blockDim.y * sizeof(F), stream>>>(in, partial_sums, width, height, op);
    
    size_t new_height = gridDim.x;
    gridDim.x = (((new_height - 1)/blockDim.x + 1) - 1) / 32 + 1;
    blockReduce<<<gridDim, blockDim, 0, stream>>>(partial_sums, result, width, new_height, op);

    CUDA_OK(cudaFreeAsync(partial_sums, stream));

    return CUDA_SUCCESS_MOON;
}


extern "C" RustCudaError vectorSumBabyBear(bb31_t* in, bb31_t* result, size_t width, size_t height, cudaStream_t stream) {
    return vectorSum(in, result, width, height, stream);
}

extern "C" RustCudaError vectorSumBabyBearExtension(bb31_extension_t* in, bb31_extension_t* result, size_t width, size_t height, cudaStream_t stream) {
    return vectorSum(in, result, width, height, stream);
}

