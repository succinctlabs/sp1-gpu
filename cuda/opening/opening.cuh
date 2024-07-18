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

template<typename F, typename EF> __global__ void shiftedPowersKernel(
    F* blockPowers, 
    EF shift, 
    Matrix<F> output, 
    size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t blockPower = blockIdx.x * blockDim.x;

    F blockGenerator = blockPowers[1]^blockPower; 

    if (idx >= n) return;

    EF outputElement =  EF(blockGenerator * blockPowers[threadIdx.x]) * shift; 
    for (size_t k = 0; k < EF::D; k++) {
        output.values[k * output.height + idx] = outputElement.value[k];
    }
}

template<typename F, typename EF> __global__ void foldEvenOddKernel(
    Matrix<F> evaluations,
    Matrix<F> inputLeaves,
    Matrix<F> output,
    Matrix<F> powers,
    F oneHalf,
    bool inputExists
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t evenIdx = 2 * idx;
    size_t oddIdx = 2 * idx + 1;

    if (idx >= output.height) return;

    EF r0Even, r0Odd, r1Even, r1Odd, evenPower, oddPower;
    for (size_t k = 0 ; k < EF::D; k++) {
        r0Even.value[k] = evaluations.values[k * evaluations.height + evenIdx];
        r1Even.value[k] = evaluations.values[(k + EF::D) * evaluations.height + evenIdx];

        r0Odd.value[k] = evaluations.values[k * evaluations.height + oddIdx];
        r1Odd.value[k] = evaluations.values[(k + EF::D) * evaluations.height + oddIdx];

        evenPower.value[k] = powers.values[k * powers.height + evenIdx];
        oddPower.value[k] = powers.values[k * powers.height + oddIdx];
    }

    EF evenValue = (oneHalf + evenPower) * r0Even + (oneHalf - evenPower) * r1Even;
    EF oddValue = (oneHalf + oddPower) * r0Odd + (oneHalf - oddPower) * r1Odd;

    for (size_t k = 0 ; k < EF::D; k++) {

        F outEven = evenValue.value[k];
        F outOdd = oddValue.value[k];

        if (inputExists) {
            outEven = outEven + inputLeaves.values[k * inputLeaves.height + idx];
            outOdd = outOdd + inputLeaves.values[(k + EF::D) * inputLeaves.height + idx];
        }

        output.values[k * output.height + idx] = outEven;
        output.values[(k + EF::D) * output.height + idx] = outOdd;

    }

}

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
    size_t* logHeights,
    bb31_extension_t* points,
    size_t* invIndices,
    bb31_extension_t* invDenoms,
    bb31_extension_t alpha,
    bb31_extension_t* alphaPowOffsets,
    bb31_extension_t* openedValues,
    size_t * openedValuesIndices,
    bb31_extension_t* reducedOpenings,
    Matrix<bb31_t>* reducedLeaves,
    size_t * heightIndices
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pointIdx = blockIdx.y * blockDim.y + threadIdx.y;

    Matrix<bb31_t> matrix = mats[pointIdx];
    size_t numRows = matrix.height;
    if (idx >= numRows) return;

    size_t invIdx = invIndices[pointIdx];
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

const size_t BLOCK_DIM = 256;
const size_t COARSE_FACTOR = 1;

const size_t MAX_LOG_HEIGHT = 32;

__global__ void ReduceSumKernel(
        size_t* logHeights, 
        size_t* invIndices,
        bb31_extension_t* reducedOpenings, 
        Matrix<bb31_t>* reducedLeaves,
        size_t * heightIndices , size_t numPoints) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    bb31_extension_t evenSums[MAX_LOG_HEIGHT];
    bb31_extension_t oddSums[MAX_LOG_HEIGHT];

    for (size_t i = 0; i < MAX_LOG_HEIGHT; i++) {
        evenSums[i] = bb31_extension_t::zero();
        oddSums[i] = bb31_extension_t::zero();
    }

    for (size_t i =0; i< numPoints; i++) {
        size_t logHeight = logHeights[i];
        size_t invIdx = invIndices[i];
        if ((2 * idx + 1) >= (1 << logHeight)) continue;
        evenSums[logHeight] += reducedOpenings[invIdx + 2 * idx];
        oddSums[logHeight]  += reducedOpenings[invIdx + 2 * idx + 1];
    }

    for (size_t h = 0; h < MAX_LOG_HEIGHT; h++) { 
        size_t heightIdx = heightIndices[h];
        Matrix<bb31_t> leafMatrix = reducedLeaves[heightIdx];

        if (idx >= leafMatrix.height) continue;
        for (size_t k = 0; k < bb31_extension_t::D; k++) {
            leafMatrix.values[k * leafMatrix.height + idx] = evenSums[h].value[k];
            leafMatrix.values[(k + bb31_extension_t::D) * leafMatrix.height + idx] = oddSums[h].value[k];
      }
    }
} 

__global__ void fetchRow(Matrix<bb31_t> matrix, size_t index, bb31_t* output) {
    for (size_t i = 0; i < matrix.width; i++) {
        output[i] = matrix.values[i * matrix.height + index];
    }
}

__global__ void fetchRowParallel(Matrix<bb31_t> matrix, size_t index, bb31_t* output) 
{
     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= matrix.width) {
        return;
     }
     output[idx] = matrix.values[idx * matrix.height + index];
}

__device__ size_t log2_ceil_usize(size_t x) {
    float log2_val = __log2f(static_cast<float>(x));
    return static_cast<size_t>(ceilf(log2_val));
}

__global__ void fetchRowTotal(
    Matrix<bb31_t> *matrix_ptr,
    size_t *matrix_idxs,
    size_t *width_offsets,
    size_t total_width,
    size_t index,
    size_t log_max_height,
    bb31_t* output
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_width) {
        return;
    }
    size_t matrix_idx = matrix_idxs[idx];
    Matrix<bb31_t> matrix = matrix_ptr[matrix_idx];
    size_t log2_height = log2_ceil_usize(matrix.height);
    size_t reduced_index = index >> (log_max_height - log2_height);
    output[idx] = matrix.values[(idx - width_offsets[matrix_idx]) * matrix.height + reduced_index];
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

constexpr size_t MAX_THREADS = 1024;

extern "C" void shiftedPowers(
    bb31_t* blockPowers, 
    bb31_extension_t shift, 
    Matrix<bb31_t> output, 
    size_t n, 
    size_t numTheads,
    size_t numBlocks) {
    opening_kernels::shiftedPowersKernel<<<numBlocks, numTheads>>>(blockPowers, shift, output, n);
}


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
    size_t numThreads = MAX_THREADS;
    size_t numBlocksX = (maxRows - 1) / numThreads + 1; 

    dim3 blockDim(numThreads);
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


extern "C" void computeReducedOpenings(
    Matrix<bb31_t>* mats,
    size_t* logHeights,
    size_t maxHeight,
    bb31_extension_t* points,
    size_t numPoints,
    size_t * invIndices,
    bb31_extension_t* invDenoms,
    bb31_extension_t alpha,
    bb31_extension_t* alphaPowOffsets,
    bb31_extension_t * openedValues,
    size_t * openedValuesIndices,
    bb31_extension_t* reducedOpenings,
    Matrix<bb31_t>* reducedLeaves,
    size_t * heightIndices
) {
    size_t numThreads = MAX_THREADS;
    size_t numBlocksX = (maxHeight - 1) / numThreads + 1; 

    dim3 blockDim(numThreads);
    dim3 gridDim(numBlocksX, numPoints);

    opening_kernels::reducedOpeningsKernel<<<gridDim, blockDim>>>(
        mats,
        logHeights,
        points,
        invIndices,
        invDenoms,
        alpha,
        alphaPowOffsets,
        openedValues,
        openedValuesIndices,
        reducedOpenings,
        reducedLeaves,
        heightIndices
    );
}

extern "C" size_t numBlocksSums(size_t maxHeight) {
    size_t numThreads = opening_kernels::BLOCK_DIM;
    size_t numBlocksX = ((maxHeight - 1) / (numThreads * opening_kernels::COARSE_FACTOR * 2)) + 1; 
    return numBlocksX;
}

extern "C" void ReduceSums(
    size_t* logHeights,
    size_t maxHeight,
    size_t * invIndices,
    bb31_extension_t* reducedOpenings,
    Matrix<bb31_t>* reducedLeaves,
    size_t * heightIndices,
    size_t numPoints
) {
    size_t numThreads = 512;
    size_t numBlocks = ((maxHeight / 2 ) - 1) / numThreads + 1;

    opening_kernels::ReduceSumKernel<<<numBlocks, numThreads>>>(
        logHeights,
        invIndices,
        reducedOpenings,
        reducedLeaves,
        heightIndices,
        numPoints
    );
}

extern "C" void fetchRow(Matrix<bb31_t> matrix, size_t index, bb31_t* output) {
#if 0
    dim3 gridDim(1);
    dim3 blockDim(1);
    opening_kernels::fetchRow<<<gridDim, blockDim>>>(matrix, index, output);
#else
    size_t blockDim = std::min(matrix.width, MAX_THREADS);
    size_t gridDim = (matrix.width - 1) / blockDim + 1;
    opening_kernels::fetchRowParallel<<<gridDim, blockDim>>>(matrix, index, output);
#endif
}

extern "C" void fetchRowTotal(
    Matrix<bb31_t> *matrix_ptr,
    size_t *matrix_idxs,
    size_t *width_offsets,
    size_t total_width, 
    size_t index, 
    size_t log_max_height,
    bb31_t* output
) {
    size_t blockDim = std::min(total_width, MAX_THREADS);
    size_t gridDim = (total_width - 1) / blockDim + 1;

    opening_kernels::fetchRowTotal<<<gridDim, blockDim>>>(
        matrix_ptr,
        matrix_idxs,
        width_offsets,
        total_width,
        index, 
        log_max_height, 
        output);
}

extern "C" void batchMultiplicativeInverse(
    bb31_extension_t* input,
    bb31_extension_t* output,
    size_t numElements
) {
    size_t numThreads = MAX_THREADS;
    size_t numBlocks = numElements / numThreads + 1;
    opening_kernels::batchMultiplicativeInverse<<<numBlocks, numThreads>>>(
        input,
        output,
        numElements
    );
}

extern "C" void foldEvenOdd(
    Matrix<bb31_t> evaluations,
    Matrix<bb31_t> inputLeaves,
    Matrix<bb31_t> output,
    Matrix<bb31_t> powers,
    bb31_t oneHalf,
    bool inputExists
) {
    size_t numThreads = MAX_THREADS;
    size_t numBlocks = (output.height - 1) / numThreads + 1;

    opening_kernels::foldEvenOddKernel<bb31_t, bb31_extension_t><<<numBlocks, numThreads>>>(
        evaluations,
        inputLeaves,
        output,
        powers,
        oneHalf,
        inputExists
    );
}

}  // namespace opening_gpu
