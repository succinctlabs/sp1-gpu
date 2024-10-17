#pragma once

#include <cuda_runtime.h>

#include "exception.cuh"
#include "moongate_cuda_cbindgen.hpp"

namespace moongate {

CudaRustError cuda_malloc(void** ptr, uintptr_t size) {
    CUDA_OK(cudaMalloc(ptr, size));
    return CUDA_SUCCESS_MOON;
}

CudaRustError cuda_malloc_host(void** ptr, uintptr_t size) {
    CUDA_OK(cudaMallocHost(ptr, size));
    return CUDA_SUCCESS_MOON;
}

CudaRustError cuda_host_register(void* ptr, uintptr_t size) {
    CUDA_OK(cudaHostRegister(ptr, size, cudaHostRegisterDefault));
    return CUDA_SUCCESS_MOON;
}

CudaRustError cuda_free(void* ptr) {
    CUDA_OK(cudaFree(ptr));
    return CUDA_SUCCESS_MOON;
}

CudaRustError cuda_free_host(void* ptr) {
    CUDA_OK(cudaFreeHost(ptr));
    return CUDA_SUCCESS_MOON;
}

CudaRustError cuda_host_unregister(void* ptr) {
    CUDA_OK(cudaHostUnregister(ptr));
    return CUDA_SUCCESS_MOON;
}

CudaRustError cuda_mem_get_info(uintptr_t* free, uintptr_t* total) {
    CUDA_OK(cudaMemGetInfo(free, total));
    return CUDA_SUCCESS_MOON;
}

CudaRustError
cuda_mem_copy_host_to_device(void* dst, const void* src, uintptr_t size) {
    CUDA_OK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return CUDA_SUCCESS_MOON;
}

CudaRustError
cuda_mem_copy_device_to_host(void* dst, const void* src, uintptr_t size) {
    CUDA_OK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return CUDA_SUCCESS_MOON;
}

CudaRustError
cuda_mem_copy_device_to_device(void* dst, const void* src, uintptr_t size) {
    CUDA_OK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    return CUDA_SUCCESS_MOON;
}

}  // namespace moongate