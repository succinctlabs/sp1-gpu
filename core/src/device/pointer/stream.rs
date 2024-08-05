use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaError, DeviceAllocator, TryAllocError},
};

use super::{CopyRawFrom, DefaultAllocatorPointer, Offset, RawDevicePointer, RawPointer};

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

impl<T: Copy> DefaultAllocatorPointer for DeviceStreamPointer<T> {
    type Allocator = CudaStream;

    fn allocator(&self) -> &Self::Allocator {
        &self.stream
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

impl<T: Copy> CopyRawFrom<DeviceStreamPointer<T>> for DeviceStreamPointer<T> {
    unsafe fn copy_raw_from(
        &mut self,
        src: &DeviceStreamPointer<T>,
        len: usize,
    ) -> Result<(), CudaError> {
        self.stream
            .cuda_memcpy_device_to_device_async(self.ptr, src.ptr, len)?;
        Ok(())
    }
}

impl<T: Copy> CopyRawFrom<*const T> for DeviceStreamPointer<T> {
    unsafe fn copy_raw_from(&mut self, src: &*const T, len: usize) -> Result<(), CudaError> {
        self.stream
            .cuda_memcpy_host_to_device_async(self.ptr, *src, len)?;
        Ok(())
    }
}

impl<T: Copy> CopyRawFrom<DeviceStreamPointer<T>> for *mut T {
    unsafe fn copy_raw_from(
        &mut self,
        src: &DeviceStreamPointer<T>,
        len: usize,
    ) -> Result<(), CudaError> {
        src.stream
            .cuda_memcpy_device_to_host_async(*self, src.ptr, len)?;
        Ok(())
    }
}

impl<T> Offset for DeviceStreamPointer<T> {
    unsafe fn add(&self, rhs: usize) -> Self {
        DeviceStreamPointer {
            ptr: self.ptr.add(rhs),
            stream: self.stream.clone(),
        }
    }
}

impl<T: Copy> RawDevicePointer for DeviceStreamPointer<T> {
    fn sync(&self) -> Result<(), CudaError> {
        self.stream.synchronize()
    }

    fn stream_raw(&self) -> *mut std::ffi::c_void {
        self.stream.raw()
    }
}
