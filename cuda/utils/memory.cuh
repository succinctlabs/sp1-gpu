#include "exception.cuh"

#include <cuda_runtime.h>

extern "C" rustCudaError_t cuda_malloc(void **devPtr, size_t size) {
    CUDA_OK(cudaMalloc(devPtr, size));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_free(void *devPtr) {
    CUDA_OK(cudaFree(devPtr));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_mem_copy_host_to_device(void *dst,
                                                        const void *src,
                                                        size_t count) {
    CUDA_OK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_mem_copy_device_to_host(void *dst,
                                                        const void *src,
                                                        size_t count) {
    CUDA_OK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_mem_copy_device_to_device(void *dst,
                                                          const void *src,
                                                          size_t count) {
    CUDA_OK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
    return CUDA_SUCCESS_MOON;
}