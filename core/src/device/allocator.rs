use std::hint;
use std::time::Duration;

use thiserror::Error;

use crate::device::error::CudaError;
use crate::device::memory::cuda_malloc;

use super::DevicePointer;

pub const DEFAULT_ALLOCATOR: DefaultDeviceAllocator = DefaultDeviceAllocator;

/// TODO: implement a thread::yield_now() when spin limits are reached.
#[allow(dead_code)]
const SPIN_LIMIT: u32 = 6;

#[derive(Debug, Clone, Error)]
#[error("Failed to allocate memory {0}")]
pub struct TryAllocError(#[from] CudaError);

#[derive(Debug, Clone, Error)]
#[error(transparent)]
pub struct AllocError(#[from] CudaError);

#[derive(Debug, Clone, Error)]
pub enum AllocTimeoutError {
    #[error("Failed to allocate memory {0}")]
    CudaError(#[from] CudaError),
    #[error("Timeout")]
    Timeout,
}

pub trait DeviceAllocator<P> {
    /// Try to allocate memory on the device.
    ///
    /// # Safety
    /// The memory returned by this function will not be freed unless explicity done by the caller.
    unsafe fn try_alloc(&self, len: usize) -> Result<P, TryAllocError>;

    /// Allocate memory on the device.
    ///
    /// This function will block until the memory is available. The method will return an error if
    /// the allocator failed for a reason other than out of memory.
    ///
    /// # Safety
    /// See [Self::try_alloc]
    unsafe fn alloc(&self, len: usize) -> Result<P, AllocError> {
        loop {
            match self.try_alloc(len) {
                Ok(ptr) => return Ok(ptr),
                Err(TryAllocError(CudaError::OutOfMemory(_))) => {
                    hint::spin_loop();
                }
                Err(TryAllocError(e)) => return Err(AllocError(e)),
            }
        }
    }

    /// Trt to allocate memory on the device or return an error after a timeout.
    ///
    /// # Safety
    /// See [Self::try_alloc]
    unsafe fn alloc_timeout(&self, len: usize, timeout: Duration) -> Result<P, AllocTimeoutError> {
        let start = std::time::Instant::now();
        loop {
            match self.try_alloc(len) {
                Ok(ptr) => return Ok(ptr),
                Err(TryAllocError(CudaError::OutOfMemory(_))) => {
                    if start.elapsed() > timeout {
                        return Err(AllocTimeoutError::Timeout);
                    }
                    hint::spin_loop();
                }
                Err(TryAllocError(e)) => return Err(AllocTimeoutError::CudaError(e)),
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DefaultDeviceAllocator;

impl<T: Copy> DeviceAllocator<DevicePointer<T>> for DefaultDeviceAllocator {
    unsafe fn try_alloc(&self, len: usize) -> Result<DevicePointer<T>, TryAllocError> {
        Ok(DevicePointer::from_raw(cuda_malloc(len)?))
    }
}
