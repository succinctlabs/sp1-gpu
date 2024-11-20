// CUDA runtime bindings.

use std::ffi::c_void;

use crate::device::error::CudaRustError;

use super::{event::CudaEventHandle, stream::CudaStreamHandle};

/// cbindgen:ignore
extern "C" {
    pub(crate) static DEFAULT_STREAM: CudaStreamHandle;

    pub(crate) fn cuda_device_synchronize() -> CudaRustError;
    pub(crate) fn cuda_event_create(event: *mut CudaEventHandle) -> CudaRustError;
    pub(crate) fn cuda_event_destroy(event: CudaEventHandle) -> CudaRustError;
    pub(crate) fn cuda_event_record(
        event: CudaEventHandle,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
    pub(crate) fn cuda_event_synchronize(event: CudaEventHandle) -> CudaRustError;
    pub(crate) fn cuda_event_elapsed_time(
        ms: *mut f32,
        start: CudaEventHandle,
        end: CudaEventHandle,
    ) -> CudaRustError;

    pub(crate) fn cuda_stream_create(stream: *mut CudaStreamHandle) -> CudaRustError;
    pub(crate) fn cuda_stream_destroy(stream: CudaStreamHandle) -> CudaRustError;
    pub(crate) fn cuda_stream_synchronize(stream: CudaStreamHandle) -> CudaRustError;

    pub(crate) fn cuda_stream_wait_event(
        stream: CudaStreamHandle,
        event: CudaEventHandle,
    ) -> CudaRustError;

    // Async memory operations.

    pub(crate) fn cuda_malloc_async(
        devPtr: *mut *mut c_void,
        size: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    pub(crate) fn cuda_mem_set_async(
        dst: *mut c_void,
        value: u8,
        size: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    pub(crate) fn cuda_free_async(devPtr: *mut c_void, stream: CudaStreamHandle) -> CudaRustError;

    pub(crate) fn cuda_mem_copy_device_to_device_async(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
    pub(crate) fn cuda_mem_copy_host_to_device_async(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
    pub(crate) fn cuda_mem_copy_device_to_host_async(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
    pub(crate) fn cuda_mem_copy_host_to_host_async(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

}
