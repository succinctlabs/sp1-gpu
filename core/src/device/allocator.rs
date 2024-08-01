use crate::device::error::CudaError;
use crate::device::memory::cuda_malloc;

use super::DevicePointer;

pub const DEFAULT_ALLOCATOR: DefaultDeviceAllocator = DefaultDeviceAllocator;

pub trait DeviceAllocator<P> {
    /// # Safety
    ///
    /// TODO
    unsafe fn alloc(&self, len: usize) -> Result<P, CudaError>;
}

#[derive(Debug, Clone, Copy)]
pub struct DefaultDeviceAllocator;

impl<T: Copy> DeviceAllocator<DevicePointer<T>> for DefaultDeviceAllocator {
    unsafe fn alloc(&self, len: usize) -> Result<DevicePointer<T>, CudaError> {
        Ok(DevicePointer::from_raw(cuda_malloc(len)?))
    }
}
