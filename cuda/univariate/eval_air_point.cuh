#pragma once

#include "../reduce/reduce.cuh"
#include "../reduce/air_point.cuh"
#include "../fields/bb31_t.cuh"
#include "../fields/bb31_extension_t.cuh"



template<typename F, typename EF> __global__ void partialUnivariateEvalAirPointKernel(
    air_point_t<EF>* partialEvaluations,
    const F* polynomialBatch,
    const F domainGenerator,
    const F domainNormalizer,
    const EF evalPoint,
    const EF vanishingPoly, 
    size_t width,
    size_t log_height) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    air_point_t<EF> thread_val = air_point_t(EF::zero(), EF::zero());

    size_t batchIdx = blockDim.y * blockIdx.y + threadIdx.y;
    if (batchIdx >= width) {
        return;
    }

    // Compute the lagrange polynomial.
    size_t height = 1U << log_height;

    // Stride loop to accumulate partial sum
     F domainPoint = domainGenerator^(blockIdx.x * blockDim.x + threadIdx.x);
     F domainPowerStride = domainGenerator^(blockDim.x * gridDim.x);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height; i += blockDim.x * gridDim.x) {
        EF largrangePolynomialLocal = vanishingPoly / (evalPoint - domainPoint);
        EF largrangePolynomialNext = vanishingPoly / (evalPoint * domainGenerator  - domainPoint);
        largrangePolynomialLocal *= domainNormalizer * domainPoint;
        largrangePolynomialNext *= domainNormalizer * domainPoint;
        F evaluation =  polynomialBatch[batchIdx * height + i]; 
        thread_val.local += largrangePolynomialLocal * evaluation;
        thread_val.next +=  largrangePolynomialNext * evaluation; 
        domainPoint *= domainPowerStride;
    }

    // Allocate shared memory
    extern __shared__ unsigned char memory[];
    air_point_t<EF>* shared = reinterpret_cast<air_point_t<EF>*>(memory);

    // Warp-level reduction within tiles
    AddOp<air_point_t<EF>> op;
    thread_val = partialBlockReduce(block, tile, thread_val, shared, op);

    // Write the result to the partial_sums array
    if (block.thread_rank() == 0) {
        partialEvaluations[batchIdx * gridDim.x + blockIdx.x] = shared[0];
    }
}

template<typename F, typename EF> RustCudaError univariateAirPointEval(
    air_point_t<EF>* result,
    const F* polynomailBatch, 
    const F domainGenerator,
    const F domainNormalizer,
    const EF evalPoint,
    const EF vanishingPoly, 
    size_t width, 
    size_t log_height,
    cudaStream_t stream) {
    size_t height = 1U << log_height;
    dim3 blockDim(512, 1, 1);
    size_t numReduceBlocks = (((height - 1)/blockDim.x + 1) - 1) / 16 + 1;
    dim3 gridDim(numReduceBlocks, width, 1);

    // Allocate the partial sums and set them to zero. 
    air_point_t<EF> * partialEvaluations;
    CUDA_OK(cudaMallocAsync(&partialEvaluations, sizeof(air_point_t<EF>) * gridDim.x * width, stream));

    size_t numTiles = blockDim.x / 32;

    partialUnivariateEvalAirPointKernel<<<gridDim, blockDim, numTiles * blockDim.y * sizeof(air_point_t<EF>), stream>>>(
        partialEvaluations,
        polynomailBatch, 
        domainGenerator,
        domainNormalizer,
        evalPoint,
        vanishingPoly, 
        width, 
        log_height);
    
    size_t new_height = gridDim.x;
    gridDim.x = (((new_height - 1)/blockDim.x + 1) - 1) / 32 + 1;
    // Initialize the result value.
    //
    // *Warning*: this assumes the zero of `F` is just given by the zero byte pattern.
    CUDA_OK(cudaMemsetAsync(result, 0, sizeof(air_point_t<EF>) * width, stream));
    // Compute the result from the partially reduced evalutations.
    AddOp<air_point_t<EF>> op;
    blockReduce<<<gridDim, blockDim, 0, stream>>>(partialEvaluations, result, width, new_height, op);
    // Free the memory used.
    CUDA_OK(cudaFreeAsync(partialEvaluations, stream));

    return CUDA_SUCCESS_MOON;
}



extern "C" RustCudaError evalUnivariateAirPointBabyBear(
    air_point_t<bb31_extension_t>* result,
    const bb31_t* polynomailBatch, 
    const bb31_t domainGenerator,
    const bb31_t domainNormalizer,
    const bb31_extension_t evalPoint,
    const bb31_extension_t vanishingPoly,  
    size_t width, 
    size_t log_height,
    cudaStream_t stream) {
    return univariateAirPointEval(
        result, 
        polynomailBatch, 
        domainGenerator, 
        domainNormalizer,
        evalPoint,
        vanishingPoly,
        width,
        log_height, 
        stream);
}