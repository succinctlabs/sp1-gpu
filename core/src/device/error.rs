use std::ffi::{c_char, CStr};

use moongate_bloc::alloc::AllocError;
use thiserror::Error;

use crate::device::ffi::CUDA_SUCCESS_MOON;

use super::ffi::CUDA_OUT_OF_MEMORY;

#[derive(Debug, Clone, Error)]
pub enum CudaError {
    #[error("Device out of memory: {0}")]
    OutOfMemory(#[from] AllocError),
    #[error("Cuda error: {0}")]
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct CudaRustError {
    pub(crate) message: *const c_char,
}

impl CudaRustError {
    pub fn to_result(self) -> Result<(), CudaError> {
        self.into()
    }
}

impl From<CudaRustError> for Result<(), CudaError> {
    fn from(value: CudaRustError) -> Self {
        unsafe {
            if value == CUDA_SUCCESS_MOON {
                Ok(())
            } else if value == CUDA_OUT_OF_MEMORY {
                Err(CudaError::OutOfMemory(AllocError))
            } else {
                Err(CudaError::Other(CStr::from_ptr(value.message).to_str().unwrap().to_string()))
            }
        }
    }
}
