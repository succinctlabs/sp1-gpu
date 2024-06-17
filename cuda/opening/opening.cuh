#include <cuda_runtime.h>

#include <ntt/ntt.cuh>

#include "../fields/bb31_extension_t.cuh"
#include "../utils/exception.cuh"
#include "../utils/matrix.cuh"

namespace opening_kernels {
__global__ void interpolateCosetStage1(
    Matrix<bb31_t> cosetEvals,
    size_t cosetHeight,
    size_t cosetLogHeight,
    bb31_t shift,
    bb31_extension_t point,
    bb31_t* gPowers,
    bb31_extension_t* output
) {
    extern __shared__ bb31_extension_t sdata[];

    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t rowStride = blockDim.y * gridDim.y;

    bb31_extension_t sum = bb31_extension_t::zero();
    for (size_t i = row; i < cosetHeight; i += rowStride) {
        size_t rev = bit_rev(i, cosetLogHeight);
        bb31_extension_t diff = point - shift * gPowers[i];
        bb31_extension_t scale = gPowers[i] * diff.reciprocal();
        sum += scale * cosetEvals.values[col * cosetEvals.width + rev];
    }

    size_t tid = threadIdx.x * blockDim.y + threadIdx.y;
    sdata[tid] = sum;
    __syncthreads();

    if (tid == 0) {
        bb31_extension_t blockSum = bb31_extension_t::zero();
        for (size_t i = 0; i < blockDim.x * blockDim.y; i++) {
            blockSum += sdata[i];
        }
        size_t gid = blockIdx.x * gridDim.y + blockIdx.y;
        output[gid] = blockSum;
    }
}

__global__ void interpolateCosetStage2(
    bb31_extension_t* partialSums,
    bb31_extension_t barycentricScalar,
    bb31_extension_t* output,
    size_t numBlocks
) {
    output[blockIdx.x] = bb31_extension_t::zero();
    for (size_t i = 0; i < numBlocks; i++) {
        output[blockIdx.x] += partialSums[blockIdx.x * numBlocks + i];
    }
    output[blockIdx.x] *= barycentricScalar;
}

__global__ void initializeReducedOpeningsForLogHeight(
    bb31_extension_t* reducedOpeningsForLogHeight,
    size_t numRows
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numRows) {
        return;
    }
    reducedOpeningsForLogHeight[idx] = bb31_extension_t::zero();
}

__global__ void computeReducedOpeningsForLogHeight(
    Matrix<bb31_t> matrix,
    bb31_extension_t* invDenoms,
    bb31_extension_t* alphaPowers,
    bb31_extension_t alphaPowOffset,
    bb31_extension_t sumAlphaPowTimesY,
    bb31_extension_t* reducedOpeningsForLogHeight
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    bb31_extension_t rowSum = bb31_extension_t::zero();
    for (size_t i = 0; i < matrix.height; i++) {
        rowSum += matrix.values[i * matrix.width + idx] * alphaPowers[i];
    }
    reducedOpeningsForLogHeight[idx] +=
        invDenoms[idx] * alphaPowOffset * (rowSum - sumAlphaPowTimesY);
}

__global__ void fetchRow(
    Matrix<bb31_t> matrix,
    size_t index,
    bb31_t *output
) {
    for (size_t i = 0; i < matrix.width; i++) {
        output[i] = matrix.values[i * matrix.height + index];
    }
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
    dim3 stage1Grid(cosetEvals.height, 32);
    dim3 stage1Block(1, 32);
    dim3 stage2Grid(cosetEvals.height);
    dim3 stage2Block(1);

    // Allocate the intermeddiate output for the first stage.
    bb31_extension_t* stage1Output;
    CUDA_UNWRAP(cudaMalloc(
        (void**)&stage1Output,
        sizeof(bb31_extension_t) * stage1Grid.x * stage1Grid.y
    ));

    // Compute the strided column-wise dot products.
    opening_kernels::interpolateCosetStage1<<<
        stage1Grid,
        stage1Block,
        sizeof(bb31_extension_t) * stage1Block.x * stage1Block.y>>>(
        cosetEvals,
        cosetHeight,
        cosetLogHeight,
        shift,
        point,
        gPowers,
        stage1Output
    );

    // Accumulate the strided sums into sums.
    opening_kernels::interpolateCosetStage2<<<stage2Grid, stage2Block>>>(
        stage1Output,
        barycentricScalar,
        output,
        stage1Grid.y
    );
 
    // Free the output from the first stage.
    CUDA_UNWRAP(cudaFree(stage1Output)); 
}

extern "C" void computeReducedOpeningForLogHeight(
    Matrix<bb31_t> matrix,
    bb31_extension_t* invDenoms,
    bb31_extension_t* alphaPowers,
    bb31_extension_t alphaPowOffset,
    bb31_extension_t sumAlphaPowTimesY,
    bb31_extension_t* reducedOpeningsForLogHeight
) {
    size_t numThreads = 32;
    size_t numBlocks = matrix.width / numThreads + 1;

    // Initialize the reduced openings for the log height.
    opening_kernels::initializeReducedOpeningsForLogHeight<<<numBlocks, numThreads>>>(
        reducedOpeningsForLogHeight,
        matrix.width
    );

    // Compute the reduced openings for the log height.
    opening_kernels::computeReducedOpeningsForLogHeight<<<matrix.width, 1>>>(
        matrix,
        invDenoms,
        alphaPowers,
        alphaPowOffset,
        sumAlphaPowTimesY,
        reducedOpeningsForLogHeight
    );
}

extern "C" void fetchRow(
    Matrix<bb31_t> matrix,
    size_t index,
    bb31_t* output
) {
    dim3 gridDim(1);
    dim3 blockDim(1);
    opening_kernels::fetchRow<<<gridDim, blockDim>>>(matrix, index, output);
}
}  // namespace opening_gpu
