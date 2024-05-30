use std::time::Duration;

use crate::{
    device::error::CudaError,
    runtime::{event::CudaEvent, ffi, stream::UnsafeCudaStream},
};

pub struct CudaInstant(pub(crate) CudaEvent);

impl CudaInstant {
    pub fn now() -> Result<Self, CudaError> {
        let default_stream = UnsafeCudaStream::default();
        default_stream.now()
    }

    pub fn elapsed(&self) -> Result<Duration, CudaError> {
        let stream = UnsafeCudaStream::default();
        let end = CudaEvent::new()?;
        stream.record(&end)?;
        stream.wait_event(&end)?;
        end.synchronize()?;
        let mut ms: f32 = 0.0;
        unsafe { ffi::cuda_event_elapsed_time(&mut ms, self.0 .0, end.0) }.to_result()?;

        let s = ms as f64 * 1e-3;
        Ok(Duration::from_secs_f64(s))
    }
}
