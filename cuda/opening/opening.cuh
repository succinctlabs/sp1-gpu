#pragma once

#include <cuda_runtime.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <ntt/ntt.cuh>

#include "../fields/bb31_extension_t.cuh"
#include "../hashes/poseidon2/poseidon2_bb31_16.cuh"
#include "../hashes/poseidon2/poseidon2_bn254_3.cuh"
#include "../matrix/matrix.cuh"
#include "../utils/exception.cuh"
#include "moongate_cuda_cbindgen.hpp"

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
    bb31_extension_t* reducedOpenings
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

__device__ size_t log2_ceil_usize(size_t x) {
    float log2_val = __log2f(static_cast<float>(x));
    return static_cast<size_t>(ceilf(log2_val));
}

__global__ void calculateOpenings(
    Matrix<bb31_t> *matrix_ptr,
    size_t *width_offsets,
    size_t *query_indices,
    size_t total_matrices,
    size_t total_width, 
    size_t total_indices,
    size_t log_max_height,
    bool is_answering,
    bb31_t* output
) {
    size_t index_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_idx >= total_indices) { return; }
    
    size_t matrix_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (matrix_idx >= total_matrices) { return; }
    Matrix<bb31_t> matrix = matrix_ptr[matrix_idx];
    
    size_t value_idx = blockIdx.z * blockDim.z + threadIdx.z;
    if (value_idx >= matrix.width) { return; }
    
    size_t index = query_indices[index_idx];
    output += index_idx * total_width + width_offsets[matrix_idx];

    size_t bits_reduced = (is_answering) ?
        (matrix_idx + 1) : 
        (log_max_height - log2_ceil_usize(matrix.height));
    output[value_idx] = matrix.values[value_idx * matrix.height + (index >> bits_reduced)];
}


template<typename HashParams>
__global__ void calculateProof(
    size_t *query_indices,
    size_t *log_max_heights,
    size_t *offsets,
    const size_t total_indices,
    const size_t total_data,
    const size_t log_global_max_height,
    const size_t sum_log_max_heights,
    typename HashParams::F_t (**digest_layers) [HashParams::DIGEST_WIDTH],
    typename HashParams::F_t (*output) [HashParams::DIGEST_WIDTH],
    bool is_answering
) {
    size_t index_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_idx >= total_indices) { return; }
    size_t index = query_indices[index_idx];
    size_t output_idx = index_idx * sum_log_max_heights;

    size_t data_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (data_idx >= total_data) { return; }

    size_t log_max_height = log_max_heights[data_idx];
    size_t offset = offsets[data_idx];
    output_idx += offset;

    size_t i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= log_max_height) { return; }

    size_t bits_reduced = (is_answering) ?
        (data_idx + 1) : 
        (log_global_max_height - log_max_height);
    size_t curr_index = index >> bits_reduced;

    for (int ii = 0; ii < HashParams::DIGEST_WIDTH; ii++)
        output[output_idx + i][ii] = digest_layers[offset + i][(curr_index >> i) ^ 1][ii];
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
};  // namespace opening_kernels

namespace opening_gpu {

constexpr size_t MAX_THREADS = 1024;

inline void shiftedPowers(
    bb31_t* blockPowers,
    bb31_extension_t shift,
    Matrix<bb31_t> output,
    size_t n,
    size_t numTheads,
    size_t numBlocks
) {
    opening_kernels::shiftedPowersKernel<<<numBlocks, numTheads>>>(
        blockPowers,
        shift,
        output,
        n
    );
}

inline void computeInverseDenominators(
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

inline void interpolateCosets(
    bb31_t** polysEvals,
    size_t numPolys,
    size_t* cosetHeights,
    size_t* cosetLogHeights,
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

inline void computeReducedOpenings(
    Matrix<bb31_t>* mats,
    size_t* logHeights,
    size_t maxHeight,
    bb31_extension_t* points,
    size_t numPoints,
    size_t* invIndices,
    bb31_extension_t* invDenoms,
    bb31_extension_t alpha,
    bb31_extension_t* alphaPowOffsets,
    bb31_extension_t* openedValues,
    size_t* openedValuesIndices,
    bb31_extension_t* reducedOpenings
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
        reducedOpenings
    );
}

inline size_t numBlocksSums(size_t maxHeight) {
    size_t numThreads = opening_kernels::BLOCK_DIM;
    size_t numBlocksX =
        ((maxHeight - 1) / (numThreads * opening_kernels::COARSE_FACTOR * 2))
        + 1;
    return numBlocksX;
}

inline void ReduceSums(
    size_t* logHeights,
    size_t maxHeight,
    size_t* invIndices,
    bb31_extension_t* reducedOpenings,
    Matrix<bb31_t>* reducedLeaves,
    size_t* heightIndices,
    size_t numPoints
) {
    size_t numThreads = 512;
    size_t numBlocks = ((maxHeight / 2) - 1) / numThreads + 1;

    opening_kernels::ReduceSumKernel<<<numBlocks, numThreads>>>(
        logHeights,
        invIndices,
        reducedOpenings,
        reducedLeaves,
        heightIndices,
        numPoints
    );
}

inline void calculateOpenings(
    Matrix<bb31_t>* matrix_ptr,
    size_t* width_offsets,
    size_t* query_indices,
    size_t total_matrices,
    size_t total_width,
    size_t max_width,
    size_t total_indices,
    size_t log_max_height,
    bool is_answering,
    bb31_t* output
) {
    // The idea of balancing thread count in blockDim based on
    // min and max possible amount of indices, matrices and width.
    // The efficient way of managing it: skipping blocks rather then
    // threads inside a block (because of WARP parallelism).
    dim3 blockDim(
        std::min(total_indices, static_cast<size_t>(32)),
        std::min(total_matrices, static_cast<size_t>(8)),
        std::min(max_width, static_cast<size_t>(4))
    );
    dim3 gridDim(
        (total_indices - 1) / blockDim.x + 1,
        (total_matrices - 1) / blockDim.y + 1,
        (max_width - 1) / blockDim.z + 1
    );

    opening_kernels::calculateOpenings<<<gridDim, blockDim>>>(
        matrix_ptr,
        width_offsets,
        query_indices,
        total_matrices,
        total_width,
        total_indices,
        log_max_height,
        is_answering,
        output
    );
}

inline void calculateProof(
    size_t* query_indices,
    size_t* log_max_heights,
    size_t* offset,
    const size_t total_indices,
    const size_t total_data,
    const size_t log_global_max_height,
    const size_t sum_log_max_heights,
    void*** digests,
    void** output,
    bool is_answering,
    size_t field_id
) {
    dim3 blockDim(
        std::min(total_indices, static_cast<size_t>(32)),
        std::min(total_data, static_cast<size_t>(1)),
        std::min(log_global_max_height, static_cast<size_t>(32))
    );
    dim3 gridDim(
        (total_indices - 1) / blockDim.x + 1,
        (total_data - 1) / blockDim.y + 1,
        (log_global_max_height - 1) / blockDim.z + 1
    );

    // If field is BabyBear
    if (field_id == 0) {
        auto typed_digests =
            reinterpret_cast<poseidon2_bb31_16::BabyBear::F_t(**)
                                 [poseidon2_bb31_16::BabyBear::DIGEST_WIDTH]>(
                digests
            );
        auto typed_output =
            reinterpret_cast<poseidon2_bb31_16::BabyBear::F_t(*)
                                 [poseidon2_bb31_16::BabyBear::DIGEST_WIDTH]>(
                output
            );
        opening_kernels::calculateProof<poseidon2_bb31_16::BabyBear>
            <<<gridDim, blockDim>>>(
                query_indices,
                log_max_heights,
                offset,
                total_indices,
                total_data,
                log_global_max_height,
                sum_log_max_heights,
                typed_digests,
                typed_output,
                is_answering
            );
    }
    // If field is Bn254
    else if (field_id == 1) {
        auto typed_digests =
            reinterpret_cast<poseidon2_bn254_3::Bn254::F_t(**)
                                 [poseidon2_bn254_3::Bn254::DIGEST_WIDTH]>(
                digests
            );
        auto typed_output =
            reinterpret_cast<poseidon2_bn254_3::Bn254::F_t(*)
                                 [poseidon2_bn254_3::Bn254::DIGEST_WIDTH]>(
                output
            );
        opening_kernels::calculateProof<poseidon2_bn254_3::Bn254>
            <<<gridDim, blockDim>>>(
                query_indices,
                log_max_heights,
                offset,
                total_indices,
                total_data,
                log_global_max_height,
                sum_log_max_heights,
                typed_digests,
                typed_output,
                is_answering
            );
    } else {
        // This is unreachable as the correct id should be passed.
        assert(false);
    }
}

inline void batchMultiplicativeInverse(
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

inline void foldEvenOdd(
    Matrix<bb31_t> evaluations,
    Matrix<bb31_t> inputLeaves,
    Matrix<bb31_t> output,
    Matrix<bb31_t> powers,
    bb31_t oneHalf,
    bool inputExists
) {
    size_t numThreads = MAX_THREADS;
    size_t numBlocks = (output.height - 1) / numThreads + 1;

    opening_kernels::foldEvenOddKernel<bb31_t, bb31_extension_t>
        <<<numBlocks, numThreads>>>(
            evaluations,
            inputLeaves,
            output,
            powers,
            oneHalf,
            inputExists
        );
}

}  // namespace opening_gpu

namespace moongate {

void shifted_powers_raw(
    const F* block_powers,
    EF shift,
    MatrixViewMutDevice<F> output,
    uintptr_t n,
    uintptr_t num_threads,
    uintptr_t num_blocks
) {
    opening_gpu::shiftedPowers(
        std::bit_cast<bb31_t*>(block_powers),
        std::bit_cast<bb31_extension_t>(shift),
        std::bit_cast<Matrix<bb31_t>>(output),
        n,
        num_threads,
        num_blocks
    );
}

void compute_inverse_denominators(
    uintptr_t max_rows,
    uintptr_t num_points,
    const uintptr_t* inv_row_indices,
    const uintptr_t* nums_rows,
    const uintptr_t* logs_num_rows,
    const F* shifts,
    const F* thread_generator_powers,
    const EF* points,
    EF* inv_denoms
) {
    opening_gpu::computeInverseDenominators(
        max_rows,
        num_points,
        std::bit_cast<size_t*>(inv_row_indices),
        std::bit_cast<size_t*>(nums_rows),
        std::bit_cast<size_t*>(logs_num_rows),
        std::bit_cast<bb31_t*>(shifts),
        std::bit_cast<bb31_t*>(thread_generator_powers),
        std::bit_cast<bb31_extension_t*>(points),
        std::bit_cast<bb31_extension_t*>(inv_denoms)
    );
}

void interpolate_cosets_raw(
    const BabyBear* const* polys_evals,
    uintptr_t num_polys,
    const uintptr_t* coset_heights,
    const uintptr_t* coset_log_heights,
    const F* shift,
    const EF* point,
    const EF* barycentric_scalars,
    const F* g_values,
    EF* output
) {
    opening_gpu::interpolateCosets(
        std::bit_cast<bb31_t**>(polys_evals),
        num_polys,
        std::bit_cast<size_t*>(coset_heights),
        std::bit_cast<size_t*>(coset_log_heights),
        std::bit_cast<bb31_t*>(shift),
        std::bit_cast<bb31_extension_t*>(point),
        std::bit_cast<bb31_extension_t*>(barycentric_scalars),
        std::bit_cast<bb31_t*>(g_values),
        std::bit_cast<bb31_extension_t*>(output)
    );
}

void compute_reduced_openings(
    const MatrixViewDevice<F>* mats,
    const uintptr_t* log_heights,
    uintptr_t max_height,
    const EF* points,
    uintptr_t num_points,
    const uintptr_t* inv_indices,
    const EF* inv_denoms,
    EF alpha,
    const EF* alpha_pow_offsets,
    const EF* ys,
    const uintptr_t* ys_indices,
    EF* reduced_openings
) {
    opening_gpu::computeReducedOpenings(
        std::bit_cast<Matrix<bb31_t>*>(mats),
        std::bit_cast<size_t*>(log_heights),
        max_height,
        std::bit_cast<bb31_extension_t*>(points),
        num_points,
        std::bit_cast<uintptr_t*>(inv_indices),
        std::bit_cast<bb31_extension_t*>(inv_denoms),
        std::bit_cast<bb31_extension_t>(alpha),
        std::bit_cast<bb31_extension_t*>(alpha_pow_offsets),
        std::bit_cast<bb31_extension_t*>(ys),
        std::bit_cast<uintptr_t*>(ys_indices),
        std::bit_cast<bb31_extension_t*>(reduced_openings)
    );
}

void reduce_sums(
    const uintptr_t* log_heights,
    uintptr_t max_height,
    const uintptr_t* inv_indices,
    const EF* reduced_openings,
    MatrixViewMutDevice<F>* reduced_openings_leaves,
    const uintptr_t* height_indices,
    uintptr_t num_points
) {
    opening_gpu::ReduceSums(
        std::bit_cast<size_t*>(log_heights),
        max_height,
        std::bit_cast<size_t*>(inv_indices),
        std::bit_cast<bb31_extension_t*>(reduced_openings),
        std::bit_cast<Matrix<bb31_t>*>(reduced_openings_leaves),
        std::bit_cast<size_t*>(height_indices),
        num_points
    );
}

uintptr_t num_block_sums(uintptr_t max_height) {
    return opening_gpu::numBlocksSums(max_height);
}

void calculate_openings(
    const MatrixViewDevice<F>* matrix_ptr,
    const uintptr_t* width_offsets,
    const uintptr_t* query_indices,
    uintptr_t total_matrices,
    uintptr_t total_width,
    uintptr_t max_width,
    uintptr_t total_indices,
    uintptr_t log_max_height,
    bool is_answering,
    F* output
) {
    opening_gpu::calculateOpenings(
        std::bit_cast<Matrix<bb31_t>*>(matrix_ptr),
        std::bit_cast<size_t*>(width_offsets),
        std::bit_cast<size_t*>(query_indices),
        total_matrices,
        total_width,
        max_width,
        total_indices,
        log_max_height,
        is_answering,
        std::bit_cast<bb31_t*>(output)
    );
}

void calculate_proofs(
    const uintptr_t* query_indices,
    const uintptr_t* log_max_heights,
    const uintptr_t* log_max_heights_offsets,
    uintptr_t total_indices,
    uintptr_t total_data,
    uintptr_t log_max_height,
    uintptr_t sum_log_max_height,
    const void* const* const* digests,
    void** output,
    bool is_answering,
    uintptr_t field_id
) {
    opening_gpu::calculateProof(
        std::bit_cast<size_t*>(query_indices),
        std::bit_cast<size_t*>(log_max_heights),
        std::bit_cast<size_t*>(log_max_heights_offsets),
        total_indices,
        total_data,
        log_max_height,
        sum_log_max_height,
        std::bit_cast<void***>(digests),
        std::bit_cast<void**>(output),
        is_answering,
        field_id
    );
}

void batch_multiplicative_inverse(
    const EF* input,
    EF* output,
    uintptr_t num_elements
) {
    opening_gpu::batchMultiplicativeInverse(
        std::bit_cast<bb31_extension_t*>(input),
        std::bit_cast<bb31_extension_t*>(output),
        num_elements
    );
}

void fold_even_odd_raw(
    MatrixViewDevice<F> evaluations,
    MatrixViewDevice<F> input_leaves,
    MatrixViewMutDevice<F> output,
    MatrixViewDevice<F> powers,
    F one_half,
    bool input_exists
) {
    opening_gpu::foldEvenOdd(
        std::bit_cast<Matrix<bb31_t>>(evaluations),
        std::bit_cast<Matrix<bb31_t>>(input_leaves),
        std::bit_cast<Matrix<bb31_t>>(output),
        std::bit_cast<Matrix<bb31_t>>(powers),
        std::bit_cast<bb31_t>(one_half),
        input_exists
    );
}

}  // namespace moongate