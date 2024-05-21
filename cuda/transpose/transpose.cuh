#include "../utils/matrix.cuh"


const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

namespace matrix_kernels {
__global__ void transpose_naive(bb31_t *output, Matrix input) {
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
}  // namespace matrix_kernels


extern "C" void transpose_naive(bb31_t *output, Matrix input) {
    dim3 dimGrid(ceil(input.height  /(double) TILE_DIM), ceil(input.width /(double) TILE_DIM), 1);
    dim3 dimBlock(BLOCK_ROWS, TILE_DIM, 1);
    matrix_kernels::transpose_naive<<<dimGrid, dimBlock>>>(output, input);
}


