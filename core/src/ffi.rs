use std::ffi::c_void;

use crate::error::CudaRustError;

extern "C" {
    pub(crate) static CUDA_SUCCESS: CudaRustError;

    pub(crate) fn cuda_malloc(ptr: *mut *mut c_void, count: usize) -> CudaRustError;

    pub(crate) fn cuda_free(ptr: *const c_void) -> CudaRustError;

    pub(crate) fn cuda_mem_copy_host_to_device(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
    ) -> CudaRustError;

    pub(crate) fn cuda_mem_copy_device_to_host(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
    ) -> CudaRustError;

    pub(crate) fn cuda_mem_copy_device_to_device(
        dst: *const c_void,
        src: *const c_void,
        count: usize,
    ) -> CudaRustError;
}
