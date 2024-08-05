use std::hint;
use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;
use std::{ffi::c_void, mem, ptr};

use thiserror::Error;

use crate::{device::error::CudaError, time::CudaInstant};

use super::{event::CudaEvent, ffi};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct CudaStreamHandle(*mut c_void);

#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct CudaStreamOwned(CudaStreamHandle);

unsafe impl Send for CudaStreamOwned {}
unsafe impl Sync for CudaStreamOwned {}

#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct CudaStream(Arc<CudaStreamOwned>);

#[derive(Debug, Clone, Error)]
pub enum AllocTimeoutError {
    #[error("Failed to allocate memory {0}")]
    CudaError(#[from] CudaError),
    #[error("Timeout")]
    Timeout,
}

impl CudaStream {
    pub fn create() -> Result<Self, CudaError> {
        let mut ptr = CudaStreamHandle(ptr::null_mut());
        unsafe { ffi::cuda_stream_create(&mut ptr as *mut CudaStreamHandle) }.to_result()?;
        Ok(Self(Arc::new(CudaStreamOwned(ptr))))
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { ffi::cuda_stream_synchronize(self.0 .0) }.to_result()
    }

    pub fn now(&self) -> Result<CudaInstant, CudaError> {
        let event = CudaEvent::new()?;
        self.record(&event)?;
        Ok(CudaInstant(event))
    }

    pub fn record(&self, event: &CudaEvent) -> Result<(), CudaError> {
        unsafe { ffi::cuda_event_record(event.0, self.0 .0) }.to_result()
    }

    pub fn elapsed(&self, start: &CudaInstant) -> Result<Duration, CudaError> {
        let end = CudaEvent::new()?;
        self.record(&end)?;
        end.synchronize()?;
        let mut ms: f32 = 0.0;
        unsafe { ffi::cuda_event_elapsed_time(&mut ms, start.0 .0, end.0) }.to_result()?;

        let s = ms as f64 * 1e-3;
        Ok(Duration::from_secs_f64(s))
    }

    pub fn wait_event(&self, event: &CudaEvent) -> Result<(), CudaError> {
        unsafe { ffi::cuda_stream_wait_event(self.0 .0, event.0) }.to_result()
    }

    /// # Safety
    ///
    /// TODO
    unsafe fn cuda_malloc_async<T: Copy>(&self, size: usize) -> Result<*mut T, CudaError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe {
            ffi::cuda_malloc_async(
                &mut ptr as *mut *mut c_void,
                size * mem::size_of::<T>(),
                self.0 .0,
            )
        }
        .to_result()?;
        Ok(ptr as *mut T)
    }

    /// # Safety
    ///
    /// TODO
    pub unsafe fn try_alloc<T: Copy>(&self, len: usize) -> Result<*mut T, CudaError> {
        self.cuda_malloc_async(len)
    }

    /// Allocate memory on the device.
    ///
    /// This function will block until the memory is available. The method will return an error if
    /// the allocator failed for a reason other than out of memory.
    ///
    /// # Safety
    /// See [Self::try_alloc]
    pub unsafe fn alloc<T: Copy>(&self, len: usize) -> Result<*mut T, CudaError> {
        loop {
            match self.try_alloc(len) {
                Ok(ptr) => return Ok(ptr),
                Err(CudaError::OutOfMemory(_)) => {
                    hint::spin_loop();
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Trt to allocate memory on the device or return an error after a timeout.
    ///
    /// # Safety
    /// See [Self::try_alloc]
    pub unsafe fn alloc_timeout<T: Copy>(
        &self,
        len: usize,
        timeout: Duration,
    ) -> Result<*mut T, AllocTimeoutError> {
        let start = std::time::Instant::now();
        loop {
            match self.try_alloc(len) {
                Ok(ptr) => return Ok(ptr),
                Err(CudaError::OutOfMemory(_)) => {
                    if start.elapsed() > timeout {
                        return Err(AllocTimeoutError::Timeout);
                    }
                    hint::spin_loop();
                }
                Err(e) => return Err(AllocTimeoutError::CudaError(e)),
            }
        }
    }

    /// # Safety
    ///
    /// TODO
    pub unsafe fn free_async<T: Copy>(&self, ptr: *mut T) -> Result<(), CudaError> {
        unsafe { ffi::cuda_free_async(ptr as *mut c_void, self.0 .0) }.to_result()
    }

    /// # Safety
    ///
    /// TODO
    pub unsafe fn cuda_memcpy_device_to_device_async<T: Copy>(
        &self,
        dst: *mut T,
        src: *const T,
        count: usize,
    ) -> Result<(), CudaError> {
        unsafe {
            ffi::cuda_mem_copy_device_to_device_async(
                dst as *mut c_void,
                src as *const c_void,
                count,
                self.0 .0,
            )
        }
        .to_result()
    }

    /// # Safety
    ///
    /// TODO
    pub unsafe fn cuda_memcpy_host_to_device_async<T: Copy>(
        &self,
        dst: *mut T,
        src: *const T,
        count: usize,
    ) -> Result<(), CudaError> {
        unsafe {
            ffi::cuda_mem_copy_host_to_device_async(
                dst as *mut c_void,
                src as *const c_void,
                count,
                self.0 .0,
            )
        }
        .to_result()
    }

    /// # Safety
    ///
    /// TODO
    pub unsafe fn cuda_memcpy_device_to_host_async<T: Copy>(
        &self,
        dst: *mut T,
        src: *const T,
        count: usize,
    ) -> Result<(), CudaError> {
        unsafe {
            ffi::cuda_mem_copy_device_to_host_async(
                dst as *mut c_void,
                src as *const c_void,
                count,
                self.0 .0,
            )
        }
        .to_result()
    }

    /// # Safety
    ///
    /// TODO
    pub unsafe fn cuda_memcpy_host_to_host_async<T: Copy>(
        &self,
        dst: *mut T,
        src: *const T,
        count: usize,
    ) -> Result<(), CudaError> {
        unsafe {
            ffi::cuda_mem_copy_host_to_host_async(
                dst as *mut c_void,
                src as *const c_void,
                count,
                self.0 .0,
            )
        }
        .to_result()
    }
}

impl Default for CudaStream {
    fn default() -> Self {
        let raw = CudaStreamOwned(unsafe { ffi::DEFAULT_STREAM });
        Self(Arc::new(raw))
    }
}

impl Drop for CudaStreamOwned {
    fn drop(&mut self) {
        if self.0 != unsafe { ffi::DEFAULT_STREAM } {
            unsafe { ffi::cuda_stream_destroy(self.0) }
                .to_result()
                .unwrap();
        }
    }
}

impl Deref for CudaStream {
    type Target = CudaStreamOwned;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use crate::device::DeviceBuffer;

    use super::*;

    #[test]
    fn test_default_stream() {
        let stream = CudaStream::default();
        let event = CudaEvent::new().unwrap();
        stream.record(&event).unwrap();

        // Get a big buffer and measure the time it takes to copy it.
        let data = vec![0u32; 1 << 22];
        let mut buffer = DeviceBuffer::<u32>::with_capacity(data.len()).unwrap();
        let time = stream.now().unwrap();
        buffer.extend_from_host_slice(&data);
        let elapsed = stream.elapsed(&time).unwrap();
        println!("{:?}", elapsed);
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_streams() {
        let stream = CudaStream::create().unwrap();

        // Get a big buffer and measure the time it takes to copy it.
        let data = vec![0u32; 1 << 22];
        let time = stream.now().unwrap();
        unsafe {
            let buf = stream.cuda_malloc_async::<u32>(data.len()).unwrap();
            stream
                .cuda_memcpy_host_to_device_async(buf, data.as_ptr(), data.len())
                .unwrap();
            stream.free_async(buf).unwrap();
            let end = CudaEvent::new().unwrap();
            stream.record(&end).unwrap();
            let elapsed = stream.elapsed(&time).unwrap();
            println!("{:?}", elapsed);
        }
    }
}
