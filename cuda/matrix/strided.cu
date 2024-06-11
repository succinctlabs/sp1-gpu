
#include "type.cuh"


template<typename T> __global__ void IntoStrided(Matrix<T> output, Matrix<T> input, size_t stride, size_t offset) {
    size_t RowIdx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (RowIdx >= output.height) {
        return;
    }

    size_t InRowIdx = (RowIdx * stride) + offset;

    for (size_t col = 0; col < output.width; col++) {
        output.values[col * output.height + RowIdx] = input.values[col * input.height + InRowIdx];
    }
}