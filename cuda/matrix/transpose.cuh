#pragma once

#include <bit>
#include <cassert>

#include "../fields/bb31_t.cuh"
#include "moongate_cuda_cbindgen.hpp"
#include "type.cuh"

namespace matrix_transpose {
const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

__global__ void TransposeNaiveRowToCol(bb31_t* output, Matrix<bb31_t> input) {
    size_t id_x = (blockIdx.x * TILE_DIM) + threadIdx.x;
    size_t id_y = (blockIdx.y * TILE_DIM) + threadIdx.y;

    size_t len = input.width * input.height;

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        size_t idx_in = (id_x + j) * input.width + id_y;
        size_t idx_out = id_y * input.height + id_x + j;
        if (idx_in < len && idx_out < len)
            output[idx_out] = input.values[idx_in];
    }
}

__global__ void TransposeNaiveColToRow(bb31_t* output, Matrix<bb31_t> input) {
    size_t id_x = (blockIdx.x * TILE_DIM) + threadIdx.x;
    size_t id_y = (blockIdx.y * TILE_DIM) + threadIdx.y;

    size_t len = input.width * input.height;

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        size_t idx_in = id_y * input.height + id_x + j;
        size_t idx_out = (id_x + j) * input.width + id_y;
        if (idx_in < len && idx_out < len)
            output[idx_out] = input.values[idx_in];
    }
}

__global__ void TransposeBlowupNaiveRowToCol(
    bb31_t* output,
    Matrix<bb31_t> input,
    size_t log_blowup
) {
    size_t id_x = (blockIdx.x * TILE_DIM) + threadIdx.x;
    size_t id_y = (blockIdx.y * TILE_DIM) + threadIdx.y;

    size_t ext_height = input.height << log_blowup;
    size_t len = input.width * input.height;
    size_t ext_len = input.width * ext_height;

#pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        size_t idx_in = (id_x + j) * input.width + id_y;
        size_t idx_out =
            id_y * ext_height + ext_height - input.height + id_x + j;
        if (idx_in < len && idx_out < ext_len)
            output[idx_out] = input.values[idx_in];
    }
}

inline void
transpose_naive(bb31_t* output, Matrix<bb31_t> input, cudaStream_t stream) {
    dim3 dimGrid(
        ceil(input.height / (double)TILE_DIM),
        ceil(input.width / (double)TILE_DIM),
        1
    );
    dim3 dimBlock(BLOCK_ROWS, TILE_DIM, 1);
    if (input.row_major) {
        TransposeNaiveRowToCol<<<dimGrid, dimBlock, 0, stream>>>(output, input);
    } else {
        TransposeNaiveColToRow<<<dimGrid, dimBlock, 0, stream>>>(output, input);
    }
}

inline void transpose_blowup_naive(
    bb31_t* output,
    Matrix<bb31_t> input,
    size_t log_blowup,
    cudaStream_t stream
) {
    assert(input.row_major);
    dim3 dimGrid(
        ceil(input.height / (double)TILE_DIM),
        ceil(input.width / (double)TILE_DIM),
        1
    );
    dim3 dimBlock(BLOCK_ROWS, TILE_DIM, 1);
    TransposeBlowupNaiveRowToCol<<<dimGrid, dimBlock, 0, stream>>>(
        output,
        input,
        log_blowup
    );
}

}  // namespace matrix_transpose

namespace moongate {
void transpose_naive(
    BabyBear* output,
    MatrixViewDevice<BabyBear> input,
    CudaStreamHandle stream
) {
    matrix_transpose::transpose_naive(
        std::bit_cast<bb31_t*>(output),
        std::bit_cast<Matrix<bb31_t>>(input),
        std::bit_cast<cudaStream_t>(stream)
    );
}

void transpose_blowup_naive(
    BabyBear* output,
    MatrixViewDevice<BabyBear> input,
    uintptr_t log_blowup,
    CudaStreamHandle stream
) {
    matrix_transpose::transpose_blowup_naive(
        std::bit_cast<bb31_t*>(output),
        std::bit_cast<Matrix<bb31_t>>(input),
        log_blowup,
        std::bit_cast<cudaStream_t>(stream)
    );
}
}  // namespace moongate
