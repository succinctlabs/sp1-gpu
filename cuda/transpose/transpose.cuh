#include "../utils/matrix.cuh"


const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

namespace matrix_kernels {
__global__ void transpose_naive(bb31_t *output, Matrix<bb31_t> input) {
    size_t id_x = (blockIdx.x * TILE_DIM) + threadIdx.x;
    size_t id_y = (blockIdx.y * TILE_DIM) + threadIdx.y;
     
    size_t len = input.width * input.height;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j+=BLOCK_ROWS) {
      size_t idx_in = (id_x + j) * input.width + id_y;
      size_t idx_out = id_y  * input.height + id_x + j;
      if (idx_in < len && idx_out < len)
        output[idx_out] = input.values[idx_in];
    }
 }

 __global__ void transpose_blowup_naive(bb31_t *output, Matrix<bb31_t> input, size_t log_blowup) {
    size_t id_x = (blockIdx.x * TILE_DIM) + threadIdx.x;
    size_t id_y = (blockIdx.y * TILE_DIM) + threadIdx.y;

    size_t ext_height = input.height << log_blowup;
    size_t len = input.width * input.height;
    size_t ext_len = input.width * ext_height;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j+=BLOCK_ROWS) {
      size_t idx_in = (id_x + j) * input.width + id_y;
      size_t idx_out = id_y  * ext_height + ext_height - input.height + id_x + j;
      if (idx_in < len && idx_out < ext_len ) 
          output[idx_out] = input.values[idx_in];
    }
 }

}  // namespace matrix_kernels


extern "C" void transpose_naive(bb31_t *output, Matrix<bb31_t> input) {
    dim3 dimGrid(ceil(input.height  /(double) TILE_DIM), ceil(input.width /(double) TILE_DIM), 1);
    dim3 dimBlock(BLOCK_ROWS, TILE_DIM, 1);
    matrix_kernels::transpose_naive<<<dimGrid, dimBlock>>>(output, input);
}

extern "C" void transpose_blowup_naive(bb31_t *output, Matrix<bb31_t> input, size_t log_blowup) {
    dim3 dimGrid(ceil(input.height  /(double) TILE_DIM), ceil(input.width /(double) TILE_DIM), 1);
    dim3 dimBlock(BLOCK_ROWS, TILE_DIM, 1);
    matrix_kernels::transpose_blowup_naive<<<dimGrid, dimBlock>>>(output, input, log_blowup);
}
