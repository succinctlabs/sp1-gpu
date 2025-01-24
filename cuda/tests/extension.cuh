
#include "../fields/bb31_extension_t.cuh"

template<typename T>
__global__ void test_extension_operations(
    T* const a,
    T* const b,
    T* add,
    T* sub,
    T* mul,
    T* div,
    size_t n
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        add[i] = a[i] + b[i];
        mul[i] = a[i] * b[i];
        sub[i] = a[i] - b[i];
        div[i] = a[i] / b[i];
    }
}

extern "C" void test_bb31_extension(
    bb31_extension_t* a,
    bb31_extension_t* b,
    bb31_extension_t* add,
    bb31_extension_t* sub,
    bb31_extension_t* mul,
    bb31_extension_t* div,
    size_t n,
    size_t block_size,
    size_t grid_size
) {
    test_extension_operations<<<grid_size, block_size>>>(
        a,
        b,
        add,
        sub,
        mul,
        div,
        n
    );
}