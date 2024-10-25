#pragma once

#include "../reduce/reduce.cuh"



template<typename F, typename EF> __global__ void partialUnivariateEvalKernel(
    EF* partialEvaluations,
    const F* polynomailBatch,
    const F domainGenerator,
    const EF evalPoint, 
    size_t width,
    size_t log_height) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    EF thread_val = EF::zero();

    size_t batchIdx = blockDim.y * blockIdx.y + threadIdx.y;
    if (batchIdx >= width) {
        return;
    }

    // Compute the lagrange polynomial.

    // First, compute `z^height`
    EF pointPower = evalPoint;
    for (size_t k = 0; k < log_height; k++) {
        pointPower *= pointPower; 
    }

    size_t height = 1U << log_height;

    // The vanishing polynomial on the domain is `z^height - 1`
    EF vanishingPoly = pointPower - EF::one();

    // The un-normalized lagrange polynomial is given by `(z^height - 1) / (z - point)`
    F domainPoint = domainGenerator^(blockIdx.x * blockDim.x + threadIdx.x);
    EF unormalizedLargrange = vanishingPoly / (evalPoint - EF(domainPoint)); 

    // Stride loop to accumulate partial sum
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height; i += blockDim.x * gridDim.x) {
        thread_val += unormalizedLargrange * polynomailBatch[batchIdx * height + i];
    }

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    F* shared = reinterpret_cast<F*>(memory);

    // Warp-level reduction within tiles
    AddOp<EF> op;
    thread_val = partialBlockReduce(block, tile, thread_val, shared, op);

    // Write the result to the partial_sums array
    if (block.thread_rank() == 0) {
        partialEvaluations[batchIdx * gridDim.x + blockIdx.x] = shared[0];
    }
}


template<typename F, typename EF> RustCudaError univariateEval(
    EF* result,
    const F* polynomailBatch, 
    const F domainGenerator,
    const EF evalPoint, 
    size_t width, 
    size_t log_height,
    cudaStream_t stream) {
    size_t height = 1U << log_height;
    dim3 blockDim(512, 1, 1);
    size_t numReduceBlocks = (((height - 1)/blockDim.x + 1) - 1) + 1;
    dim3 gridDim(numReduceBlocks, width, 1);

    // Allocate the partial sums and set them to zero. 
    F * partialEvaluations;
    CUDA_OK(cudaMallocAsync(&partialEvaluations, sizeof(F) * gridDim.x * width, stream));

    size_t numTiles = blockDim.x / 32;

    partialUnivariateEvalKernel<<<gridDim, blockDim, numTiles * blockDim.y * sizeof(F), stream>>>(
        partialEvaluations,
        polynomailBatch, 
        domainGenerator,
        evalPoint, 
        width, 
        log_height);
    
    size_t new_height = gridDim.x;
    gridDim.x = (((new_height - 1)/blockDim.x + 1) - 1) / 32 + 1;
    AddOp<F> op;
    blockReduce<<<gridDim, blockDim, 0, stream>>>(partialEvaluations, result, width, new_height, op);

    CUDA_OK(cudaFreeAsync(partialEvaluations, stream));

    return CUDA_SUCCESS_MOON;
}
