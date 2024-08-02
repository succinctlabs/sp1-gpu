use crate::device::{
    error::CudaError,
    memory::{copy_device_to_device, copy_device_to_host, copy_host_to_device, cuda_free},
};

use super::{CopyRawFrom, RawPointer};

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct DevicePointer<T>(*mut T);

impl<T> RawPointer for DevicePointer<T> {
    type Data = T;

    fn as_ptr(&self) -> *const T {
        self.0
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.0
    }

    fn free(&mut self) {
        unsafe { cuda_free(self.0) }.unwrap()
    }
}

impl<T> DevicePointer<T> {
    pub const fn from_raw(ptr: *mut T) -> Self {
        Self(ptr)
    }
}

impl<T: Copy> CopyRawFrom<DevicePointer<T>> for DevicePointer<T> {
    unsafe fn copy_from(&mut self, src: &DevicePointer<T>, len: usize) -> Result<(), CudaError> {
        copy_device_to_device(self.0, src.0, len)
    }
}

impl<T: Copy> CopyRawFrom<*mut T> for DevicePointer<T> {
    unsafe fn copy_from(&mut self, src: &*mut T, len: usize) -> Result<(), CudaError> {
        copy_host_to_device(self.0, *src, len)
    }
}

impl<T: Copy> CopyRawFrom<DevicePointer<T>> for *mut T {
    unsafe fn copy_from(&mut self, src: &DevicePointer<T>, len: usize) -> Result<(), CudaError> {
        copy_device_to_host(*self, src.0, len)
    }
}
