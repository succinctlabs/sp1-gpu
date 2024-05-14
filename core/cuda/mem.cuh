

#include <cuda_runtime.h>
#include "exception.cuh"

extern "C" rustCudaError_t cuda_malloc(void **devPtr, size_t size)
{
    CUDA_OK(cudaMalloc(devPtr, size));
    return CUDA_SUCCESS;
}

extern "C" rustCudaError_t cuda_free(void *devPtr)
{
    CUDA_OK(cudaFree(devPtr));
    return CUDA_SUCCESS;
}