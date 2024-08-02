use std::{ffi::c_void, ptr};

use super::ffi;
use crate::device::error::CudaError;

#[repr(transparent)]
pub struct CudaEvent(pub(crate) *mut c_void);

impl CudaEvent {
    pub fn new() -> Result<Self, CudaError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe { ffi::cuda_event_create(&mut ptr as *mut *mut c_void) }.to_result()?;
        Ok(Self(ptr))
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { ffi::cuda_event_synchronize(self.0) }.to_result()
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe { ffi::cuda_event_destroy(self.0) }
            .to_result()
            .unwrap();
    }
}
