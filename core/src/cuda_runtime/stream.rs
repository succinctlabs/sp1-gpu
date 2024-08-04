use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;
use std::{ffi::c_void, mem, ptr};

use crate::{device::error::CudaError, time::CudaInstant};

use super::{event::CudaEvent, ffi};

#[derive(Debug)]
#[repr(transparent)]
pub struct CudaStreamHandle(*mut c_void);

unsafe impl Send for CudaStreamHandle {}
unsafe impl Sync for CudaStreamHandle {}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct CudaStream(Arc<CudaStreamHandle>);

impl CudaStream {
    pub fn create() -> Result<Self, CudaError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe { ffi::cuda_stream_create(&mut ptr as *mut *mut c_void) }.to_result()?;
        Ok(Self(Arc::new(CudaStreamHandle(ptr))))
    }

    pub(crate) fn raw(&self) -> *mut c_void {
        self.0 .0
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
    pub unsafe fn cuda_malloc_async<T: Copy>(&self, size: usize) -> Result<*mut T, CudaError> {
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
    pub unsafe fn cuda_free_async<T: Copy>(&self, ptr: *mut T) -> Result<(), CudaError> {
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
        let raw = CudaStreamHandle(unsafe { ffi::DEFAULT_STREAM });
        Self(Arc::new(raw))
    }
}

impl Drop for CudaStreamHandle {
    fn drop(&mut self) {
        if self.0 != unsafe { ffi::DEFAULT_STREAM } {
            unsafe { ffi::cuda_stream_destroy(self.0) }
                .to_result()
                .unwrap();
        }
    }
}

impl Deref for CudaStream {
    type Target = CudaStreamHandle;

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
            stream.cuda_free_async(buf).unwrap();
            let end = CudaEvent::new().unwrap();
            stream.record(&end).unwrap();
            let elapsed = stream.elapsed(&time).unwrap();
            println!("{:?}", elapsed);
        }
    }
}
