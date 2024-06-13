#include "../fields/bb31_extension_t.cuh"
#include "../utils/matrix.cuh"

#include <ntt/ntt.cuh>

namespace opening_kernels {
__global__ void interpolateCosetStage1(Matrix<bb31_t> cosetEvals,
                                       size_t cosetHeight,
                                       size_t cosetLogHeight, bb31_t shift,
                                       bb31_extension_t point, bb31_t *gPowers,
                                       bb31_extension_t *output) {
    extern __shared__ bb31_extension_t sdata[];

    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t rowStride = blockDim.y * gridDim.y;

    bb31_extension_t sum = bb31_extension_t::zero();
    for (size_t i = row; i < cosetHeight; i += rowStride) {
        size_t rev = bit_rev(i, cosetLogHeight);
        bb31_extension_t diff = point - shift * gPowers[rev];
        bb31_extension_t scale = gPowers[rev] * diff.reciprocal();
        sum += cosetEvals.values[col * cosetEvals.height + rev] * scale;
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[col * gridDim.x + blockIdx.x] = sdata[0];
    }
}

__global__ void interpolateCosetStage2(bb31_extension_t *partialSums,
                                       bb31_extension_t *output,
                                       size_t numBlocks) {
    size_t col = blockIdx.x;
    bb31_extension_t sum = bb31_extension_t::zero();
    for (int i = 0; i < numBlocks; i++) {
        sum += partialSums[col * numBlocks + i];
    }
    output[col] = sum;
}
}  // namespace opening_kernels

namespace opening_gpu {
extern "C" void interpolateCoset(Matrix<bb31_t> cosetEvals, size_t cosetHeight,
                                 size_t cosetLogHeight, bb31_t shift,
                                 bb31_extension_t point, bb31_t *gPowers,
                                 bb31_extension_t *output, size_t numBlocksX,
                                 size_t numBlocksY, size_t numThreadsPerBlockX,
                                 size_t numThreadsPerBlockY) {
    dim3 dimBlock(numThreadsPerBlockX, numThreadsPerBlockY);
    dim3 dimGrid(numBlocksX, numBlocksY);
    opening_kernels::interpolateCosetStage1<<<dimGrid, dimBlock>>>(
        cosetEvals, cosetHeight, cosetLogHeight, shift, point, gPowers, output);

    cudaDeviceSynchronize();

    // opening_kernels::interpolateCosetStage2<<<dimGrid, dimBlock>>>(
    //     output, output, numBlocksX * numBlocksY);

    // cudaDeviceSynchronize();
}
}  // namespace opening_gpu

// struct MatrixOpenings {
//     bb31_extension_t *points;
//     size_t numPoints;
// };

// struct Round {
//     Matrix<bb31_t> *matrices;
//     MatrixOpenings *openings;
//     size_t numMatrices;
// };

// __device__ size_t log2_strict_usize(size_t x) {
//     size_t result = 0;
//     while (x > 1) {
//         x >>= 1;
//         ++result;
//     }
//     return result;
// }

// __global__ void reduceRows(Matrix<bb31_t> matrix,
//                            bb31_extension_t *reducedOpeningForLogHeight,
//                            bb31_extension_t *invDenoms, bb31_extension_t
//                            alpha, bb31_extension_t sumAlphaPowTimesY) {
//     size_t row = blockIdx.x * blockDim.x + threadIdx.x;

//     bb31_extension_t rowSum = bb31_extension_t::zero();
//     bb31_extension_t alphaPow = alpha;
//     for (size_t i = 0; i < matrix.width; i++) {
//         rowSum += matrix.values[i * matrix.height + row] * alphaPow;
//         alphaPow *= alpha;
//     }

//     reducedOpeningForLogHeight[row] +=
//         invDenoms[row] * (rowSum - sumAlphaPowTimesY);
// }

// __device__ void open(Round *rounds, size_t numRounds, bb31_extension_t alpha,
// size_t globalMaxWidth, size_t globalMaxHeight, size_t logGlobalMaxHeight) {
//     size_t global_max_width = 0;
//     for (size_t i = 0; i < numRounds; i++) {
//         Round round = rounds[i];
//         for (size_t j = 0; j < round.numMatrices; j++) {
//             Matrix<bb31_t> matrix = round.matrices[j];
//             if (matrix.width > global_max_width) {
//                 global_max_width = matrix.width;
//             }
//         }
//     }

//     size_t global_max_height = 0;
//     global_max_width = global_max_width * numRounds;
//     for (size_t i = 0; i < numRounds; i++) {
//         Round round = rounds[i];
//         for (size_t j = 0; j < round.numMatrices; j++) {
//             Matrix<bb31_t> matrix = round.matrices[j];
//             if (matrix.height > global_max_height) {
//                 global_max_height = matrix.height;
//             }
//         }
//     }
// }