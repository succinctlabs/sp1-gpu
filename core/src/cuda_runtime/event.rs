use std::{ffi::c_void, ptr, sync::Arc};

use super::ffi;
use crate::device::error::CudaError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CudaEventHandle(*mut c_void);

#[derive(Debug)]
#[repr(transparent)]
pub struct CudaEventOwned(CudaEventHandle);

unsafe impl Send for CudaEventOwned {}
unsafe impl Sync for CudaEventOwned {}

#[derive(Debug, Clone)]
pub struct CudaEvent(Arc<CudaEventOwned>);

impl CudaEvent {
    pub fn new() -> Result<Self, CudaError> {
        let mut ptr = CudaEventHandle(ptr::null_mut());
        unsafe { ffi::cuda_event_create(&mut ptr as *mut CudaEventHandle) }.to_result()?;
        Ok(Self(Arc::new(CudaEventOwned(ptr))))
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { ffi::cuda_event_synchronize(self.0 .0) }.to_result()
    }

    #[inline]
    pub fn handle(&self) -> CudaEventHandle {
        self.0 .0
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe { ffi::cuda_event_destroy(self.handle()) }.to_result().unwrap();
    }
}
