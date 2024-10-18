#pragma once

// CUDA runtime bindings.

#include <cuda.h>
#include <nvtx3/nvToolsExt.h>

#include "exception.cuh"
#include "moongate_cuda_cbindgen.hpp"


// These two functions currently have no Rust bindings.

// Create an nvtx domain.
extern "C" nvtxDomainHandle_t nvtxDomainCreateARust(char* name) {
    return nvtxDomainCreateA(name);
}

// Destroy an nvtx domain.
extern "C" void nvtxDomainDestroyARust(nvtxDomainHandle_t domain) {
    nvtxDomainDestroy(domain);
}

namespace moongate {
// Create a global nvtx range.
NvtxRangeId nvtx_range_start(const char* name){
    return nvtxRangeStart(name);
}

// Destroy a global nvtx range.
void nvtx_range_end(NvtxRangeId domain) {
    nvtxRangeEnd(domain);
}


// Sync device
 CudaRustError cuda_device_synchronize() {
    CUDA_OK(cudaDeviceSynchronize());
    return CUDA_SUCCESS_MOON;
}

// Cuda events.

 CudaRustError cuda_event_create(CudaStreamHandle* event) {
    CUDA_OK(cudaEventCreate((cudaEvent_t*)event));
    return CUDA_SUCCESS_MOON;
}

 CudaRustError cuda_event_destroy(CudaStreamHandle event) {
    CUDA_OK(cudaEventDestroy((cudaEvent_t)event));
    return CUDA_SUCCESS_MOON;
}

 CudaRustError
cuda_event_record(CudaStreamHandle event, CudaStreamHandle stream) {
    CUDA_OK(cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream));
    return CUDA_SUCCESS_MOON;
}

 CudaRustError cuda_event_synchronize(CudaStreamHandle event) {
    CUDA_OK(cudaEventSynchronize((cudaEvent_t)event));
    return CUDA_SUCCESS_MOON;
}

 CudaRustError
cuda_event_elapsed_time(float* ms, CudaStreamHandle start, CudaStreamHandle end) {
    CUDA_OK(cudaEventElapsedTime(ms, (cudaEvent_t)start, (cudaEvent_t)end));
    return CUDA_SUCCESS_MOON;
}

// Cuda streams.

extern "C" const CudaStreamHandle DEFAULT_STREAM = cudaStreamDefault;

 CudaRustError cuda_stream_create(CudaStreamHandle* stream) {
    CUDA_OK(cudaStreamCreateWithFlags((cudaStream_t*)stream, cudaStreamNonBlocking));
    return CUDA_SUCCESS_MOON;
}

 CudaRustError cuda_stream_destroy(CudaStreamHandle stream) {
    CUDA_OK(cudaStreamDestroy((cudaStream_t)stream));
    return CUDA_SUCCESS_MOON;
}

 CudaRustError cuda_stream_synchronize(CudaStreamHandle stream) {
    CUDA_OK(cudaStreamSynchronize((cudaStream_t)stream));
    return CUDA_SUCCESS_MOON;
}

 CudaRustError
cuda_stream_wait_event(CudaStreamHandle stream, CudaEventHandle event) {
    CUDA_OK(cudaStreamWaitEvent((cudaStream_t)stream,(cudaEvent_t) event));
    return CUDA_SUCCESS_MOON;
}

// Async memory operations.

 CudaRustError
cuda_malloc_async(void** devPtr, size_t size, CudaStreamHandle stream) {
    CUDA_OK(cudaMallocAsync(devPtr, size, (cudaStream_t)stream));
    return CUDA_SUCCESS_MOON;
}

 CudaRustError cuda_free_async(void* devPtr, CudaStreamHandle stream) {
    CUDA_OK(cudaFreeAsync(devPtr, (cudaStream_t)stream));
    return CUDA_SUCCESS_MOON;
}

 CudaRustError cuda_mem_copy_device_to_device_async(
    void* dst,
    const void* src,
    size_t count,
    CudaStreamHandle stream
) {
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
    return CUDA_SUCCESS_MOON;
}

 CudaRustError cuda_mem_copy_host_to_device_async(
    void* dst,
    const void* src,
    size_t count,
    CudaStreamHandle stream
) {
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, (cudaStream_t)stream));
    return CUDA_SUCCESS_MOON;
}

 CudaRustError cuda_mem_copy_device_to_host_async(
    void* dst,
    const void* src,
    size_t count,
    CudaStreamHandle stream
) {
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
    return CUDA_SUCCESS_MOON;
}

 CudaRustError cuda_mem_copy_host_to_host_async(
    void* dst,
    const void* src,
    size_t count,
    CudaStreamHandle stream
) {
    CUDA_OK(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToHost, (cudaStream_t)stream));
    return CUDA_SUCCESS_MOON;
}


}  // namespace moongate
