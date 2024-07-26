use std::ffi::c_void;

use crate::device::error::CudaRustError;

extern "C" {
    pub(crate) static CUDA_SUCCESS_MOON: CudaRustError;

    pub(crate) fn cuda_malloc(ptr: *mut *mut c_void, count: usize) -> CudaRustError;

    pub(crate) fn cuda_free(ptr: *const c_void) -> CudaRustError;

    pub(crate) fn cuda_mem_get_info(free: *mut usize, total: *mut usize) -> CudaRustError;

    pub(crate) fn cuda_malloc_host(ptr: *mut *mut c_void, count: usize) -> CudaRustError;
    pub(crate) fn cuda_host_register(ptr: *const c_void, count: usize) -> CudaRustError;
    pub(crate) fn cuda_free_host(ptr: *const c_void) -> CudaRustError;
    pub(crate) fn cuda_host_unregister(ptr: *const c_void) -> CudaRustError;

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
