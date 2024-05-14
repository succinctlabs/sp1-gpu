

#include <cuda_runtime.h>

extern "C" cudaError_t cuda_malloc(void **devPtr, size_t size)
{
    return cudaMalloc(devPtr, size);
}