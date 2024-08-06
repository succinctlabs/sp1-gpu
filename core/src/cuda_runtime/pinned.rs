use std::mem::ManuallyDrop;

use crate::device::{
    error::CudaError,
    memory::{
        cuda_free_host, cuda_host_register, cuda_host_unregister, cuda_malloc_host, ToDevice,
    },
    DeviceBuffer,
};

use super::stream::CudaStream;

#[derive(Debug)]
pub enum PinBuffer<T> {
    CudaOwnned(ManuallyDrop<Vec<T>>),
    CudaRegistered(Vec<T>),
}

impl<T> PinBuffer<T> {
    pub fn with_capacity(capacity: usize) -> Result<Self, CudaError> {
        let vec = unsafe {
            let ptr = cuda_malloc_host(capacity)?;
            Vec::from_raw_parts(ptr, 0, capacity)
        };

        Ok(Self::CudaOwnned(ManuallyDrop::new(vec)))
    }

    pub fn register(vec: Vec<T>) -> Result<Self, CudaError> {
        // Register the underlying buffer as pinned for cuda usage.
        unsafe { cuda_host_register(vec.as_ptr(), vec.capacity()) }?;
        Ok(Self::CudaRegistered(vec))
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::CudaOwnned(vec) => vec.len(),
            Self::CudaRegistered(vec) => vec.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::CudaOwnned(vec) => vec.is_empty(),
            Self::CudaRegistered(vec) => vec.is_empty(),
        }
    }
}

impl<T> Drop for PinBuffer<T> {
    fn drop(&mut self) {
        match self {
            Self::CudaOwnned(vec) => unsafe {
                cuda_free_host(vec.as_ptr()).unwrap();
            },
            Self::CudaRegistered(vec) => unsafe {
                cuda_host_unregister(vec.as_ptr()).unwrap();
            },
        }
    }
}

impl<T: Copy> ToDevice for PinBuffer<T> {
    type DeviceType = DeviceBuffer<T>;

    fn to_device_async(&self, stream: &CudaStream) -> Result<Self::DeviceType, CudaError> {
        match self {
            Self::CudaOwnned(vec) => vec.to_device_async(stream),
            Self::CudaRegistered(vec) => vec.to_device_async(stream),
        }
    }
}
