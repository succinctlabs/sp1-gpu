#pragma once

// CUDA runtime bindings.

#include <cuda.h>
#include "exception.cuh"


// Sync device 
extern "C" rustCudaError_t cuda_device_synchronize() {
    CUDA_OK(cudaDeviceSynchronize());
    return CUDA_SUCCESS_MOON; 
} 

// Cuda events.

extern "C" rustCudaError_t cuda_event_create(cudaEvent_t *event) {
    CUDA_OK(cudaEventCreate(event));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_event_destroy(cudaEvent_t event) {
    CUDA_OK(cudaEventDestroy(event));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_event_record(cudaEvent_t event, cudaStream_t stream) {
    CUDA_OK(cudaEventRecord(event, stream));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_event_synchronize(cudaEvent_t event) {
    CUDA_OK(cudaEventSynchronize(event));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_event_elapsed_time(float *ms, cudaEvent_t start, cudaEvent_t end) {
    CUDA_OK(cudaEventElapsedTime(ms, start, end));
    return CUDA_SUCCESS_MOON;
}


// Cuda streams.

extern "C" const cudaStream_t DEFAULT_STREAM = cudaStreamDefault;

extern "C" rustCudaError_t cuda_stream_create(cudaStream_t *stream) {
    CUDA_OK(cudaStreamCreate(stream));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_stream_destroy(cudaStream_t stream) {
    CUDA_OK(cudaStreamDestroy(stream));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_stream_synchronize(cudaStream_t stream) {
    CUDA_OK(cudaStreamSynchronize(stream));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_stream_wait_event(cudaStream_t stream, cudaEvent_t event) {
    CUDA_OK(cudaStreamWaitEvent(stream, event));
    return CUDA_SUCCESS_MOON;
}

// Async memory operations.

extern "C" rustCudaError_t cuda_malloc_async(void **devPtr, size_t size, cudaStream_t stream) {
    CUDA_OK(cudaMallocAsync(devPtr, size, stream));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_free_async(void *devPtr, cudaStream_t stream) {
    CUDA_OK(cudaFreeAsync(devPtr, stream));
    return CUDA_SUCCESS_MOON;
}


extern "C" rustCudaError_t cuda_mem_copy_device_to_device_async(void *dst, const void *src, size_t count, cudaStream_t stream) {
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_mem_copy_host_to_device_async(void *dst, const void *src, size_t count, cudaStream_t stream) {
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_mem_copy_device_to_host_async(void *dst, const void *src, size_t count, cudaStream_t stream) {
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t cuda_mem_copy_host_to_host_async(void *dst, const void *src, size_t count, cudaStream_t stream) {
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToHost, stream));
    return CUDA_SUCCESS_MOON;
}

