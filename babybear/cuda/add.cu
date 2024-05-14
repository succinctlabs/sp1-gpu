
#include "kernels.cuh"
#include <cuda_runtime.h>
#include "exception.cuh"

extern "C" void add_baby_bear(bb31_t *a, bb31_t *b, bb31_t *c, int n)
{
    int size = n * sizeof(bb31_t);

    bb31_t *a_d, *b_d, *c_d;

    // Allocate slices for a, b, c on the device.
    CUDA_OK(cudaMalloc((void **)&a_d, size));
    CUDA_OK(cudaMalloc((void **)&b_d, size));
    CUDA_OK(cudaMalloc((void **)&c_d, size));

    // Copy the input.
    CUDA_OK(cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice));

    // Perform the addition with a kernel invocation.
    vecAddKernel<<<ceil(n / 256.0), 256>>>(a_d, b_d, c_d, n);

    // Copy the output
    CUDA_OK(cudaMemcpy(c, c_d, size, cudaMemcpyDeviceToHost));

    // Free the device memory.
    CUDA_OK(cudaFree(a_d));
    CUDA_OK(cudaFree(b_d));
    CUDA_OK(cudaFree(c_d));
}