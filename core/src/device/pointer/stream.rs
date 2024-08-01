use crate::{
    cuda_runtime::stream::CudaStream,
    device::{DeviceAllocator, TryAllocError},
};

use super::RawPointer;

pub struct DeviceStreamPointer<T> {
    ptr: *mut T,
    stream: CudaStream,
}

impl<T> DeviceStreamPointer<T> {
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

impl<T: Copy> RawPointer for DeviceStreamPointer<T> {
    type Data = T;

    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    fn free(&mut self) {
        unsafe { self.stream.cuda_free_async(self.ptr).unwrap() }
    }
}

impl<T: Copy> DeviceAllocator<DeviceStreamPointer<T>> for CudaStream {
    unsafe fn try_alloc(&self, len: usize) -> Result<DeviceStreamPointer<T>, TryAllocError> {
        let ptr = self.cuda_malloc_async(len)?;
        Ok(DeviceStreamPointer {
            ptr,
            stream: self.clone(),
        })
    }
}
