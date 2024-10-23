#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
#include "../fields/bb31_t.cuh"

namespace cg = cooperative_groups;

//cuda::atomic<int, cuda::thread_scope_block>& total_sum


__device__ void block_sum_bb31(
    const bb31_t* A, 
    int count,
    cuda::atomic<bb31_t, cuda::thread_scope_block>& total_sum) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    bb31_t thread_sum = bb31_t::zero();

    // Stride loop over all values, each thread accumulates its part of the array.
    for (int i = block.thread_rank(); i < count; i += block.size()) {
        thread_sum += A[i];
    }

    // reduce thread sums across the tile, add the result to the atomic
    // cg::plus<int> allows cg::reduce() to know it can use hardware acceleration for addition
      cg::reduce_update_async(tile, total_sum, thread_sum, cg::plus<bb31_t>());

    // synchronize the block, to ensure all async reductions are ready
    block.sync();
}

__device__ void block_sum(
    const int* A, 
    int count,
    cuda::atomic<int, cuda::thread_scope_block>& total_sum) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int thread_sum = 0;

    // Stride loop over all values, each thread accumulates its part of the array.
    for (int i = block.thread_rank(); i < count; i += block.size()) {
        thread_sum += A[i];
    }

    // reduce thread sums across the tile, add the result to the atomic
    // cg::plus<int> allows cg::reduce() to know it can use hardware acceleration for addition
      cg::reduce_update_async(tile, total_sum, thread_sum, cg::plus<int>());

    // synchronize the block, to ensure all async reductions are ready
    block.sync();
}