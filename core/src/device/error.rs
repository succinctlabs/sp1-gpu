use std::{
    error::Error,
    ffi::{c_char, CStr},
    fmt::Display,
};

use crate::device::ffi::{CUDA_OUT_OF_MEMORY, CUDA_SUCCESS_MOON};

#[derive(Debug, Clone)]

pub enum CudaError {
    OutOfMemory(String),
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
                Err(CudaError::OutOfMemory(
                    CStr::from_ptr(value.message).to_str().unwrap().to_string(),
                ))
            } else {
                Err(CudaError::Other(
                    CStr::from_ptr(value.message).to_str().unwrap().to_string(),
                ))
            }
        }
    }
}

impl Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::OutOfMemory(msg) => write!(f, "{}", msg),
            CudaError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl Error for CudaError {}
