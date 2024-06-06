use crate::device::error::CudaError;

pub mod event;
pub(crate) mod ffi;
pub mod stream;

pub fn sync_device() -> Result<(), CudaError> {
    unsafe { ffi::cuda_device_synchronize() }.into()
}

pub fn sync_default_stream() -> Result<(), CudaError> {
    unsafe { ffi::cuda_stream_synchronize(ffi::DEFAULT_STREAM) }.into()
}
