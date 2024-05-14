use std::ffi::c_void;

use crate::error::CudaRustError;

extern "C" {
    pub(crate) static CUDA_SUCCESS: CudaRustError;

    pub(crate) fn cuda_malloc(ptr: *mut *mut c_void, size: usize) -> CudaRustError;

    pub(crate) fn cuda_free(ptr: *const c_void) -> CudaRustError;
}
