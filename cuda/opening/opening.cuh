#include <cuda_runtime.h>
#include <cstdio>
#include <ntt/ntt.cuh>

#include "../fields/bb31_extension_t.cuh"
#include "../utils/exception.cuh"
#include "../matrix/matrix.cuh"


// dim3 stage1Grid(cosetEvals.width, 32);
// dim3 stage1Block(1, 32);

namespace opening_kernels {

__global__ void interpolateCosetStage1(
    Matrix<bb31_t> cosetEvals,          
    size_t cosetHeight,                 // = ROWS (= 65536)
    size_t cosetLogHeight,              // = log2(cosetHeight) = 16
    bb31_t shift,                       // = ROOT_OF_UNITY = bb31_t(1) << (cosetLogHeight - 1)
    bb31_extension_t point,
    bb31_t* gPowers,
    bb31_extension_t* output
) {
    extern __shared__ bb31_extension_t sdata[]; // 32

    size_t col = blockIdx.x * blockDim.x + threadIdx.x; // [0..11)
    size_t row = blockIdx.y * blockDim.y + threadIdx.y; // [0..32) * 32 + [0..32) = [0..1024)
    size_t rowStride = blockDim.y * gridDim.y;          // 1024

    bb31_extension_t sum = bb31_extension_t::zero();
    for (size_t i = row; i < cosetHeight; i += rowStride) { // 0..65536
        size_t rev = bit_rev(i, cosetLogHeight);
        bb31_extension_t diff = point - shift * gPowers[i];
        bb31_extension_t scale = gPowers[i] * diff.reciprocal();
        sum += scale * cosetEvals.values[col * cosetEvals.height + rev];
    }

    size_t tid = threadIdx.x * blockDim.y + threadIdx.y;
    sdata[tid] = sum;
    __syncthreads();

    if (tid == 0) {
        bb31_extension_t blockSum = bb31_extension_t::zero();
        for (size_t i = 0; i < blockDim.x * blockDim.y; i++) { // 32
            blockSum += sdata[i];
        }
        size_t gid = blockIdx.x * gridDim.y + blockIdx.y;   // [0..11) * 32 + [0..32) = [0..32*11)
        output[gid] = blockSum;
    }
}


// dim3 stage2Grid(cosetEvals.width);
// dim3 stage2Block(1);

__global__ void interpolateCosetStage2(
    bb31_extension_t* partialSums,
    bb31_extension_t barycentricScalar,
    bb31_extension_t* output,
    size_t numBlocks                    // = 32
) {
    output[blockIdx.x] = bb31_extension_t::zero();
    for (size_t i = 0; i < numBlocks; i++) {        // [0..11)* 32 + [0..32) = [0..32*11)
        output[blockIdx.x] += partialSums[blockIdx.x * numBlocks + i];
    }
    output[blockIdx.x] *= barycentricScalar;
}


// dim3 stageGrid(cosetEvals.width);
// dim3 stageBlock(32, 32);

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

#if 0
    int steps = 64;

    extern __shared__ bb31_extension_t sdata[];
    sdata[row] = sum;
    __syncthreads();
    for (unsigned int s = rowStride/2; s >= steps; s >>= 1) {
        if (row < s) {
            sdata[row] += sdata[row + s];
        }
        __syncthreads();
    }
//#else
    extern __shared__ bb31_t sdata31_t[];
    ((bb31_extension_t*)sdata31_t)[row] = sum;
    __syncthreads();
    sdata31_t[row] += sdata31_t[row + rowStride] + sdata31_t[row + 2 * rowStride] + sdata31_t[row + 3 * rowStride];
    __syncthreads();
    for (unsigned int s = rowStride/2; s >= bb31_extension_t::D * steps; s >>= 1) {
        if (row < s) {
            sdata31_t[row] += sdata31_t[row + s];
        }
        __syncthreads();
    }


    if (row == 0) {
        bb31_extension_t blockSum = bb31_extension_t::zero();
        for (size_t i = 0; i < steps; i++) { 
            blockSum += sdata[i];   //((bb31_extension_t*)sdata31_t)[i];
        }
        output[col] = blockSum /*((bb31_extension_t*)sdata)[0]*/ * barycentricScalar;
    }
#endif
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
    for (size_t i = 0; i < matrix.width; i++) {
        rowSum += matrix.values[i * matrix.height + idx] * alphaPowers[i];
    }
    reducedOpeningsForLogHeight[idx] +=
        invDenoms[idx] * alphaPowOffset * (rowSum - sumAlphaPowTimesY);
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
#if 1
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
#else
    dim3 stage1Grid(cosetEvals.width, 32);
    dim3 stage1Block(1, 32);
    dim3 stage2Grid(cosetEvals.width);
    dim3 stage2Block(1);

    // Allocate the intermeddiate output for the first stage.
    bb31_extension_t* stage1Output;
    CUDA_UNWRAP(cudaMalloc(
        (void**)&stage1Output,
        sizeof(bb31_extension_t) * stage1Grid.x * stage1Grid.y
    ));
	// cudaDeviceSynchronize();
    // std::vector<uint8_t> h_point(sizeof(bb31_extension_t));
    // cudaMemcpy(h_point.data(), &point, sizeof(bb31_extension_t), cudaMemcpyDeviceToHost);
    // for (size_t i = 0; i < h_point.size(); i++) {
    //     printf("%d ", h_point[i]);
    // }
    // printf("\n");
    // printf("sizeof(bb31_extension_t) = %d\n", sizeof(bb31_extension_t));    // 16
    // printf("cosetEvals.width: %d, cosetHeight  %d, cosetLogHeight %d\n", cosetEvals.width, cosetHeight, cosetLogHeight);
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

    // cudaDeviceSynchronize();
    // std::vector<uint8_t> h_stage1Output(sizeof(bb31_extension_t) * stage1Grid.x * stage1Grid.y);
    // cudaMemcpy(h_stage1Output.data(), stage1Output, sizeof(bb31_extension_t) * stage1Grid.x * stage1Grid.y, cudaMemcpyDeviceToHost);
    // for (size_t i = 0; i < h_stage1Output.size(); i++) {
    //     printf("%d ", h_stage1Output[i]);
    // }
    // printf("\n\n\n");

    // Accumulate the strided sums into sums.
    opening_kernels::interpolateCosetStage2<<<stage2Grid, stage2Block>>>(
        stage1Output,
        barycentricScalar,
        output,
        stage1Grid.y
    );

    // Free the output from the first stage.
    CUDA_UNWRAP(cudaFree(stage1Output));
#endif
}

extern "C" void computeReducedOpeningForLogHeight(
    Matrix<bb31_t> matrix,
    bb31_extension_t* invDenoms,
    bb31_extension_t* alphaPowers,
    bb31_extension_t alphaPowOffset,
    bb31_extension_t sumAlphaPowTimesY,
    bb31_extension_t* reducedOpeningsForLogHeight
) {
#if 1
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
#else
    size_t numThreads = 1024;
    // TODO: ceil(matrix.height / numThreads)
    // size_t numBlocks = ceil(matrix.height /(float) numThreads);
    size_t numBlocks = matrix.height / numThreads + 1;

    // Initialize the reduced openings for the log height.
    opening_kernels::
        initializeReducedOpeningsForLogHeight<<<numBlocks, numThreads>>>(
            reducedOpeningsForLogHeight,
            matrix.height
        );

    // Compute the reduced openings for the log height.
    opening_kernels::computeReducedOpeningsForLogHeight<<<matrix.height, 1>>>(
        matrix,
        invDenoms,
        alphaPowers,
        alphaPowOffset,
        sumAlphaPowTimesY,
        reducedOpeningsForLogHeight
    );
#endif
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
