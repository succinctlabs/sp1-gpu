use std::{ffi::c_void, ptr};

use super::ffi;
use crate::device::error::CudaError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CudaEventHandle(*mut c_void);

#[repr(transparent)]
pub struct CudaEvent(CudaEventHandle);

impl CudaEvent {
    pub fn new() -> Result<Self, CudaError> {
        let mut ptr = CudaEventHandle(ptr::null_mut());
        unsafe { ffi::cuda_event_create(&mut ptr as *mut CudaEventHandle) }.to_result()?;
        Ok(Self(ptr))
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { ffi::cuda_event_synchronize(self.0) }.to_result()
    }

    #[inline]
    pub const fn handle(&self) -> CudaEventHandle {
        self.0
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe { ffi::cuda_event_destroy(self.0) }.to_result().unwrap();
    }
}
