#include <cuda_runtime.h>
#include <cstdio>
#include <ntt/ntt.cuh>

#include "../fields/bb31_extension_t.cuh"
#include "../utils/exception.cuh"
#include "../matrix/matrix.cuh"

namespace opening_kernels {

__global__ void interpolateCosetKernel(
    Matrix<bb31_t> cosetEvals,          
    size_t cosetHeight,                 // = ROWS (= 65536)
    size_t cosetLogHeight,              // = log2(cosetHeight) = 16
    bb31_t shift,                       // = ROOT_OF_UNITY = bb31_t(1) << (cosetLogHeight - 1)
    bb31_extension_t point,
    bb31_t* gPowers,
    bb31_extension_t barycentricScalar,
    bb31_extension_t* output
) {
    size_t col = blockIdx.x;                             // [0..11)
    size_t row = threadIdx.y * blockDim.x + threadIdx.x; // [0..32) * 32 + [0..32) = [0..1024)
    size_t rowStride = blockDim.x * blockDim.y;          // 1024

    bb31_extension_t sum = bb31_extension_t::zero();
    for (int i = row; i < cosetHeight; i += rowStride) { // 0..65536
        size_t rev = bit_rev(i, cosetLogHeight);
        bb31_t gPowers_i = gPowers[i];
        bb31_extension_t diff = point - shift * gPowers_i;
        bb31_extension_t scale = gPowers_i * diff.reciprocal();
        sum += scale * cosetEvals.values[col * cosetEvals.height + rev];
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
            output[col] = blockSum * barycentricScalar;
        }
    }
}



__global__ void reducedOpeningsForLogHeightKernel(
    Matrix<bb31_t> matrix,
    size_t numRows,
    bb31_extension_t* invDenoms,
    bb31_extension_t* alphaPowers,
    bb31_extension_t alphaPowOffset,
    bb31_extension_t sumAlphaPowTimesY,
    bb31_extension_t* reducedOpeningsForLogHeight
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRows) return;
    reducedOpeningsForLogHeight[idx] = bb31_extension_t::zero();
    bb31_extension_t rowSum = bb31_extension_t::zero();
    for (size_t i = 0; i < matrix.width; i++) {
        rowSum += matrix.values[i * matrix.height + idx] * alphaPowers[i];
    }
    reducedOpeningsForLogHeight[idx] +=
        invDenoms[idx] * alphaPowOffset * (rowSum - sumAlphaPowTimesY);
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
extern "C" void interpolateCoset(
    Matrix<bb31_t> cosetEvals,
    size_t cosetHeight,
    size_t cosetLogHeight,
    bb31_t shift,
    bb31_extension_t point,
    bb31_extension_t barycentricScalar,
    bb31_t* gPowers,
    bb31_extension_t* output
) {
    dim3 stageGrid(cosetEvals.width);
    dim3 stageBlock(32, 32);

    opening_kernels::interpolateCosetKernel<<<
    stageGrid, 
    stageBlock, 
    sizeof(bb31_extension_t) * stageBlock.x * stageBlock.y>>>(
        cosetEvals,
        cosetHeight,
        cosetLogHeight,
        shift,
        point,
        gPowers,
        barycentricScalar,
        output
    );
}

extern "C" void computeReducedOpeningForLogHeight(
    Matrix<bb31_t> matrix,
    bb31_extension_t* invDenoms,
    bb31_extension_t* alphaPowers,
    bb31_extension_t alphaPowOffset,
    bb31_extension_t sumAlphaPowTimesY,
    bb31_extension_t* reducedOpeningsForLogHeight
) {
    size_t numThreads = 1024;
    size_t numBlocks = (matrix.height - 1) / numThreads + 1;

    opening_kernels::reducedOpeningsForLogHeightKernel<<<numBlocks, numThreads>>>(
        matrix,
        matrix.height,
        invDenoms,
        alphaPowers,
        alphaPowOffset,
        sumAlphaPowTimesY,
        reducedOpeningsForLogHeight
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
