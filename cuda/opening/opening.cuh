#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <ntt/ntt.cuh>

#include "../fields/bb31_extension_t.cuh"
#include "../hashes/poseidon2/poseidon2_bb31_16.cuh"
#include "../hashes/poseidon2/poseidon2_bn254_3.cuh"
#include "../matrix/matrix.cuh"
#include "../utils/exception.cuh"

namespace helpers {
template<typename F>
__device__ __forceinline__ F twoAdicCosetZerofier(size_t log_n, F shift, F x) {
    F x_pow = x.exp_power_of_two(log_n);
    F shift_pow = shift.exp_power_of_two(log_n);
    F res = x_pow - shift_pow;
    return res;
}
}  // namespace helpers

namespace opening_kernels {

template<typename F, typename EF>
__global__ void
shiftedPowersKernel(F* blockPowers, EF shift, Matrix<F> output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t blockPower = blockIdx.x * blockDim.x;

    F blockGenerator = blockPowers[1] ^ blockPower;

    if (idx >= n)
        return;

    EF outputElement = EF(blockGenerator * blockPowers[threadIdx.x]) * shift;
    for (size_t k = 0; k < EF::D; k++) {
        output.values[k * output.height + idx] = outputElement.value[k];
    }
}

template<typename F, typename EF>
__global__ void foldEvenOddKernel(
    Matrix<F> evaluations,
    Matrix<F> inputLeaves,
    Matrix<F> output,
    Matrix<F> powers,
    F oneHalf,
    EF beta_squared,
    bool inputExists
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    size_t evenIdx = 2 * idx;
    size_t oddIdx = 2 * idx + 1;

    if (idx >= output.height)
        return;

    EF r0Even, r0Odd, r1Even, r1Odd, evenPower, oddPower, inputEven, inputOdd;
    for (size_t k = 0; k < EF::D; k++) {
        r0Even.value[k] = evaluations.values[k * evaluations.height + evenIdx];
        r1Even.value[k] =
            evaluations.values[(k + EF::D) * evaluations.height + evenIdx];

        r0Odd.value[k] = evaluations.values[k * evaluations.height + oddIdx];
        r1Odd.value[k] =
            evaluations.values[(k + EF::D) * evaluations.height + oddIdx];

        evenPower.value[k] = powers.values[k * powers.height + evenIdx];
        oddPower.value[k] = powers.values[k * powers.height + oddIdx];

        if (inputExists) {
            inputEven.value[k] = inputLeaves.values[k * inputLeaves.height + idx];
            inputOdd.value[k] = inputLeaves.values[(k + EF::D) * inputLeaves.height + idx];
        }
    }

    EF evenValue =
        (oneHalf + evenPower) * r0Even + (oneHalf - evenPower) * r1Even;
    EF oddValue = (oneHalf + oddPower) * r0Odd + (oneHalf - oddPower) * r1Odd;
    if (inputExists) {
        evenValue += beta_squared * inputEven;
        oddValue += beta_squared * inputOdd;
    }

    for (size_t k = 0; k < EF::D; k++) {
        output.values[k * output.height + idx] = evenValue.value[k];
        output.values[(k + EF::D) * output.height + idx] = oddValue.value[k];
    }
}

__device__ size_t log2_ceil_usize(size_t x) {
    float log2_val = __log2f(static_cast<float>(x));
    return static_cast<size_t>(ceilf(log2_val));
}

__global__ void calculateOpenings(
    Matrix<bb31_t>* matrix_ptr,
    size_t* width_offsets,
    size_t* query_indices,
    size_t total_matrices,
    size_t total_width,
    size_t total_indices,
    size_t log_max_height,
    bool is_answering,
    bb31_t* output
) {
    size_t index_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_idx >= total_indices) {
        return;
    }

    size_t matrix_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (matrix_idx >= total_matrices) {
        return;
    }
    Matrix<bb31_t> matrix = matrix_ptr[matrix_idx];

    size_t value_idx = blockIdx.z * blockDim.z + threadIdx.z;
    if (value_idx >= matrix.width) {
        return;
    }

    size_t index = query_indices[index_idx];
    output += index_idx * total_width + width_offsets[matrix_idx];

    size_t bits_reduced = (is_answering)
        ? (matrix_idx + 1)
        : (log_max_height - log2_ceil_usize(matrix.height));
    output[value_idx] =
        matrix.values[value_idx * matrix.height + (index >> bits_reduced)];
}

template<typename HashParams>
__global__ void calculateProof(
    size_t* query_indices,
    size_t* log_max_heights,
    size_t* offsets,
    const size_t total_indices,
    const size_t total_data,
    const size_t log_global_max_height,
    const size_t sum_log_max_heights,
    typename HashParams::F_t (**digest_layers)[HashParams::DIGEST_WIDTH],
    typename HashParams::F_t (*output)[HashParams::DIGEST_WIDTH],
    bool is_answering
) {
    size_t index_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_idx >= total_indices) {
        return;
    }
    size_t index = query_indices[index_idx];
    size_t output_idx = index_idx * sum_log_max_heights;

    size_t data_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (data_idx >= total_data) {
        return;
    }

    size_t log_max_height = log_max_heights[data_idx];
    size_t offset = offsets[data_idx];
    output_idx += offset;

    size_t i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= log_max_height) {
        return;
    }

    size_t bits_reduced = (is_answering)
        ? (data_idx + 1)
        : (log_global_max_height - log_max_height);
    size_t curr_index = index >> bits_reduced;

    for (int ii = 0; ii < HashParams::DIGEST_WIDTH; ii++)
        output[output_idx + i][ii] =
            digest_layers[offset + i][(curr_index >> i) ^ 1][ii];
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

extern "C" void shiftedPowers(
    bb31_t* blockPowers,
    bb31_extension_t shift,
    Matrix<bb31_t> output,
    size_t n,
    size_t numTheads,
    size_t numBlocks,
    cudaStream_t stream
) {
    opening_kernels::shiftedPowersKernel<<<numBlocks, numTheads, 0, stream>>>(
        blockPowers,
        shift,
        output,
        n
    );
}

extern "C" void calculateOpenings(
    Matrix<bb31_t>* matrix_ptr,
    size_t* width_offsets,
    size_t* query_indices,
    size_t total_matrices,
    size_t total_width,
    size_t max_width,
    size_t total_indices,
    size_t log_max_height,
    bool is_answering,
    bb31_t* output,
    cudaStream_t stream
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

    opening_kernels::calculateOpenings<<<gridDim, blockDim, 0, stream>>>(
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

extern "C" void calculateProof(
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
    size_t field_id,
    cudaStream_t stream
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
            <<<gridDim, blockDim, 0, stream>>>(
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
            <<<gridDim, blockDim, 0, stream>>>(
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

extern "C" void foldEvenOdd(
    Matrix<bb31_t> evaluations,
    Matrix<bb31_t> inputLeaves,
    Matrix<bb31_t> output,
    Matrix<bb31_t> powers,
    bb31_t oneHalf,
    bb31_extension_t beta_squared,
    bool inputExists,
    cudaStream_t stream
) {
    size_t numThreads = MAX_THREADS;
    size_t numBlocks = (output.height - 1) / numThreads + 1;

    opening_kernels::foldEvenOddKernel<bb31_t, bb31_extension_t>
        <<<numBlocks, numThreads, 0, stream>>>(
            evaluations,
            inputLeaves,
            output,
            powers,
            oneHalf,
            beta_squared,
            inputExists
        );
}

}  // namespace opening_gpu