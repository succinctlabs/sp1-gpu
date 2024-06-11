
#include "type.cuh"
#include <cassert>
#include "../fields/bb31_t.cuh"


namespace matrix_strided {
  const int TILE_DIM = 32;
  const int BLOCK_ROWS = 8;

  template<typename T> __global__ void RowStrided(Matrix<T> output, Matrix<T> input, size_t stride, size_t offset) {

    size_t Idx = (blockIdx.x * TILE_DIM) + threadIdx.x;
    size_t Idy = (blockIdx.y * TILE_DIM) + threadIdx.y;

    size_t len = output.width * output.height;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j+=BLOCK_ROWS) {
      size_t InRowIdx = (Idx + j) * stride + offset;
      size_t OutRowIdx = Idx + j;

      size_t InIdx = Idy * input.height + InRowIdx;
      size_t OutIdx = Idy * output.height + OutRowIdx;
      if (OutIdx < len)
        output.values[OutIdx] = input.values[InIdx];
    }
  }

    template<typename T> __global__ void SplitRowsNaive(Matrix<T>* outputs, Matrix<T> input, size_t stride) {

    size_t Idx = (blockIdx.x * TILE_DIM) + threadIdx.x;
    size_t Idy = (blockIdx.y * TILE_DIM) + threadIdx.y;
 
    size_t len = outputs[0].width * outputs[0].height;

    #pragma unroll
    for (size_t j = 0; j < TILE_DIM; j+=BLOCK_ROWS) {
      size_t InRowIdx = (Idx + j) * stride;
      size_t OutRowIdx = Idx + j;

      size_t InIdx = Idy * input.height + InRowIdx;
      size_t OutIdx = Idy * outputs[0].height + OutRowIdx;
      if (OutIdx < len) 
        for (size_t k =0; k < stride; k++) 
          outputs[k].values[OutIdx] = input.values[InIdx + k];
    }
  }

  extern "C" void strided_matrix(Matrix<bb31_t> output, Matrix<bb31_t> input, size_t stride, size_t offset) {
    dim3 dimGrid(ceil(output.height  /(double) TILE_DIM), ceil(output.width /(double) TILE_DIM), 1);
    dim3 dimBlock(BLOCK_ROWS, TILE_DIM, 1);
    assert(!input.row_major);
    assert(!output.row_major);
    RowStrided<<<dimGrid, dimBlock>>>(output, input, stride, offset);
 }

  extern "C" void split_rows(Matrix<bb31_t>* outputs, Matrix<bb31_t> input, size_t stride) {
    dim3 dimGrid(ceil(outputs[0].height  /(double) TILE_DIM), ceil(outputs[0].width /(double) TILE_DIM), 1);
    dim3 dimBlock(BLOCK_ROWS, TILE_DIM, 1);
    assert(!input.row_major);
    assert(!outputs[0].row_major);
    SplitRowsNaive<<<dimGrid, dimBlock>>>(outputs, input, stride);
 }
}  // namespace matrix_strided