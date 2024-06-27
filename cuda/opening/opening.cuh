#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <ntt/ntt.cuh>
#include <cstdint>

#include "../fields/bb31_extension_t.cuh"
#include "../utils/exception.cuh"
#include "../matrix/matrix.cuh"


namespace helpers {
    template<typename F> __device__ __forceinline__ F twoAdicCosetZerofier(size_t log_n, F shift, F x)  {
        F x_pow = x.exp_power_of_two(log_n);
        F shift_pow = shift.exp_power_of_two(log_n);
        F res = x_pow - shift_pow;
        return res;
    }
}

namespace opening_kernels {

__global__ void computeInverseDenominatorsKernel(
    size_t* invRowIndices,
    size_t* numsRows,
    size_t* logsNumRows,
    bb31_t* shifts,
    bb31_t* threadGeneratorPowers,
    bb31_extension_t* points,
    bb31_extension_t* invDenoms
) {
    size_t rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pointIdx = blockIdx.y * blockDim.y + threadIdx.y;

    bb31_t shift = shifts[pointIdx];
    size_t numRows = numsRows[pointIdx];
    size_t InvIdx = invRowIndices[pointIdx];

    if (rowIdx >= numRows) {
        return;
    }

    bb31_t generator = threadGeneratorPowers[pointIdx * blockDim.x + 1];
    bb31_t blockGenerator = generator^(blockIdx.x * blockDim.x);
    bb31_t genPower = blockGenerator * threadGeneratorPowers[pointIdx * blockDim.x + threadIdx.x];
    bb31_t x = shift * genPower;

    bb31_extension_t point = points[pointIdx];
    bb31_extension_t diff = bb31_extension_t(x) - point;

    size_t logNumRows = logsNumRows[pointIdx];
    size_t bitrev = bit_rev(rowIdx, logNumRows); 

    invDenoms[InvIdx + bitrev] = diff.reciprocal();
}

__global__ void interpolateCosetsKernel(
    bb31_t** polysEvals,          
    size_t* cosetHeights,                 
    size_t* cosetLogHeights,             
    bb31_t* shifts,                       
    bb31_extension_t* points,
    bb31_t* gValues,
    bb31_extension_t * barycentricScalars,
    bb31_extension_t* output
) {
    size_t index = blockIdx.x;                             
    uint32_t row = threadIdx.y * blockDim.x + threadIdx.x; 
    uint32_t rowStride = blockDim.x * blockDim.y;         

    bb31_t* polyEvals = polysEvals[index];
    size_t cosetHeight = cosetHeights[index];
    size_t cosetLogHeight = cosetLogHeights[index];
    bb31_t shift = shifts[index];
    bb31_extension_t point = points[index];
    bb31_t* gWarpPowers = gValues + index * blockDim.x;
    bb31_extension_t barycentricScalar = barycentricScalars[index];

    bb31_extension_t sum = bb31_extension_t::zero();

    bb31_t g = gWarpPowers[1];
    bb31_t gStride = g^rowStride; 
    bb31_t gPowers_i = (g^(threadIdx.y * blockDim.x)) * gWarpPowers[threadIdx.x];
    for (int i = row; i < cosetHeight; i += rowStride) { 
        size_t rev = bit_rev(i, cosetLogHeight);
        bb31_extension_t diff = point - shift * gPowers_i;
        bb31_extension_t scale = gPowers_i * diff.reciprocal();
        sum += scale * polyEvals[rev];
        gPowers_i *= gStride;
    }

    extern __shared__ bb31_extension_t sdata[];
    sdata[row] = sum;
    __syncthreads();

    int steps = 32;
    if (row < steps) {
        bb31_extension_t blockSum = bb31_extension_t::zero();
        for (int i = row; i < rowStride; i+=steps) { 
            blockSum += sdata[i];
        }
        sdata[row] = blockSum;
        __syncwarp();

        if (row == 0) {
            blockSum = bb31_extension_t::zero();
            for (int i = 0; i < steps; i++) { 
                blockSum += sdata[i];
            }
            output[index] = blockSum * barycentricScalar;
        }
    }
}


__global__ void reducedOpeningsKernel(
    Matrix<bb31_t>* mats,
    bb31_extension_t* points,
    size_t* invIndices,
    bb31_extension_t* invDenoms,
    bb31_extension_t alpha,
    bb31_extension_t* alphaPowOffsets,
    bb31_extension_t* openedValues,
    size_t * openedValuesIndices,
    bb31_extension_t* reducedOpenings
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pointIdx = blockIdx.y * blockDim.y + threadIdx.y;

    Matrix<bb31_t> matrix = mats[pointIdx];
    size_t numRows = matrix.height;
    if (idx >= numRows) return;

    size_t invIdx = invIndices[pointIdx];
    bb31_extension_t point = points[pointIdx];
    bb31_extension_t alphaPowOffset = alphaPowOffsets[pointIdx];
    size_t openValuesIdx = openedValuesIndices[pointIdx];

    bb31_extension_t rowSum = bb31_extension_t::zero();

    bb31_extension_t alphaPower = bb31_extension_t::one();
    for (size_t i = 0; i < matrix.width; i++) {
        rowSum += (matrix.values[i * matrix.height + idx] - openedValues[openValuesIdx + i]) * alphaPower;
        alphaPower *= alpha;
    }
    reducedOpenings[invIdx + idx] = invDenoms[invIdx + idx] * alphaPowOffset * rowSum;
}

__global__ void reduce(size_t * heights, size_t* invIndices, bb31_extension_t* reducedOpenings) {
} 

__global__ void reducedOpeningsForLogHeightKernel(
    Matrix<bb31_t> matrix,
    size_t numRows,
    bb31_extension_t point,
    bb31_extension_t* invDenoms,
    bb31_extension_t alpha,
    bb31_extension_t alphaPowOffset,
    bb31_extension_t* openedValues,
    bb31_extension_t* reducedOpeningsForLogHeight
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRows) return;
    reducedOpeningsForLogHeight[idx] = bb31_extension_t::zero();
    bb31_extension_t rowSum = bb31_extension_t::zero();

    bb31_extension_t alphaPower = bb31_extension_t::one();
    for (size_t i = 0; i < matrix.width; i++) {
        rowSum += (matrix.values[i * matrix.height + idx] - openedValues[i]) * alphaPower;
        alphaPower *= alpha;
    }
    reducedOpeningsForLogHeight[idx] +=
        invDenoms[idx] * alphaPowOffset * rowSum;
}

__global__ void fetchRow(Matrix<bb31_t> matrix, size_t index, bb31_t* output) {
    for (size_t i = 0; i < matrix.width; i++) {
        output[i] = matrix.values[i * matrix.height + index];
    }
}

__global__ void batchMultiplicativeInverse(
    bb31_extension_t* input,
    bb31_extension_t* output,
    size_t numElements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements) {
        return;
    }
    output[idx] = input[idx].reciprocal();
}
}  // namespace opening_kernels

namespace opening_gpu {


extern "C" void computeInverseDenominators(
    size_t maxRows,
    size_t numPoints,
    size_t* invRowIndices,
    size_t* numsRows,
    size_t* logsNumRows,
    bb31_t* shifts,
    bb31_t* threadGeneratorPowers,
    bb31_extension_t* points,
    bb31_extension_t* invDenoms
) {
    size_t numThreads = 1024;
    size_t numBlocksX = (maxRows - 1) / numThreads + 1; 

    dim3 blockDim(1024);
    dim3 gridDim(numBlocksX, numPoints);

    opening_kernels::computeInverseDenominatorsKernel<<<gridDim, blockDim>>>(
        invRowIndices,
        numsRows,
        logsNumRows,
        shifts,
        threadGeneratorPowers,
        points,
        invDenoms
    );
}

extern "C" void interpolateCosets(
    bb31_t** polysEvals,
    size_t numPolys,
    size_t*  cosetHeights,
    size_t * cosetLogHeights,
    bb31_t* shifts,
    bb31_extension_t* points,
    bb31_extension_t* barycentricScalars,
    bb31_t* gValues,
    bb31_extension_t* output
) {
    dim3 stageGrid(numPolys);
    dim3 stageBlock(32, 32);

    opening_kernels::interpolateCosetsKernel<<<
    stageGrid, 
    stageBlock, 
    sizeof(bb31_extension_t) * stageBlock.x * stageBlock.y>>>(
        polysEvals,
        cosetHeights,
        cosetLogHeights,
        shifts,
        points,
        gValues,
        barycentricScalars,
        output
    );
}


extern "C" void computeReducedOpeningForLogHeight(
    Matrix<bb31_t> matrix,
    bb31_extension_t point,
    bb31_extension_t* invDenoms,
    bb31_extension_t alpha,
    bb31_extension_t alphaPowOffset,
    bb31_extension_t * openedValues,
    bb31_extension_t* reducedOpeningsForLogHeight
) {
    size_t numThreads = 1024;
    size_t numBlocks = (matrix.height - 1) / numThreads + 1;

    opening_kernels::reducedOpeningsForLogHeightKernel<<<numBlocks, numThreads>>>(
        matrix,
        matrix.height,
        point,
        invDenoms,
        alpha,
        alphaPowOffset,
        openedValues,
        reducedOpeningsForLogHeight
    );
}

extern "C" void computeReducedOpenings(
    Matrix<bb31_t>* mats,
    size_t maxHeight,
    bb31_extension_t* points,
    size_t numPoints,
    size_t * invIndices,
    bb31_extension_t* invDenoms,
    bb31_extension_t alpha,
    bb31_extension_t* alphaPowOffsets,
    bb31_extension_t * openedValues,
    size_t * openedValuesIndices,
    bb31_extension_t* reducedOpenings
) {
    size_t numThreads = 1024;
    size_t numBlocksX = (maxHeight - 1) / numThreads + 1; 

    dim3 blockDim(1024);
    dim3 gridDim(numBlocksX, numPoints);

    opening_kernels::reducedOpeningsKernel<<<gridDim, blockDim>>>(
        mats,
        points,
        invIndices,
        invDenoms,
        alpha,
        alphaPowOffsets,
        openedValues,
        openedValuesIndices,
        reducedOpenings
    );
}

extern "C" void fetchRow(Matrix<bb31_t> matrix, size_t index, bb31_t* output) {
    dim3 gridDim(1);
    dim3 blockDim(1);
    opening_kernels::fetchRow<<<gridDim, blockDim>>>(matrix, index, output);
}

extern "C" void batchMultiplicativeInverse(
    bb31_extension_t* input,
    bb31_extension_t* output,
    size_t numElements
) {
    size_t numThreads = 1024;
    size_t numBlocks = numElements / numThreads + 1;
    opening_kernels::batchMultiplicativeInverse<<<numBlocks, numThreads>>>(
        input,
        output,
        numElements
    );
}
}  // namespace opening_gpu
