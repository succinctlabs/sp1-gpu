// CUDA runtime bindings.

use std::ffi::c_void;

use crate::device::error::CudaRustError;

extern "C" {
    pub(crate) static DEFAULT_STREAM: *mut c_void;

    pub(crate) fn cuda_device_synchronize() -> CudaRustError;
    pub(crate) fn cuda_event_create(event: *mut *mut c_void) -> CudaRustError;
    pub(crate) fn cuda_event_destroy(event: *mut c_void) -> CudaRustError;
    pub(crate) fn cuda_event_record(event: *mut c_void, stream: *mut c_void) -> CudaRustError;
    pub(crate) fn cuda_event_synchronize(event: *mut c_void) -> CudaRustError;
    pub(crate) fn cuda_event_elapsed_time(
        ms: *mut f32,
        start: *mut c_void,
        end: *mut c_void,
    ) -> CudaRustError;

    pub(crate) fn cuda_stream_create(stream: *mut *mut c_void) -> CudaRustError;
    pub(crate) fn cuda_stream_destroy(stream: *mut c_void) -> CudaRustError;
    pub(crate) fn cuda_stream_synchronize(stream: *mut c_void) -> CudaRustError;

    pub(crate) fn cuda_stream_wait_event(stream: *mut c_void, event: *mut c_void) -> CudaRustError;

    // Async memory operations.

    pub(crate) fn cuda_malloc_async(
        devPtr: *mut *mut c_void,
        size: usize,
        stream: *mut c_void,
    ) -> CudaRustError;
    pub(crate) fn cuda_free_async(devPtr: *mut c_void, stream: *mut c_void) -> CudaRustError;

    pub(crate) fn cuda_mem_copy_device_to_device_async(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        stream: *mut c_void,
    ) -> CudaRustError;
    pub(crate) fn cuda_mem_copy_host_to_device_async(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        stream: *mut c_void,
    ) -> CudaRustError;
    pub(crate) fn cuda_mem_copy_device_to_host_async(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        stream: *mut c_void,
    ) -> CudaRustError;
    pub(crate) fn cuda_mem_copy_host_to_host_async(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        stream: *mut c_void,
    ) -> CudaRustError;

}
