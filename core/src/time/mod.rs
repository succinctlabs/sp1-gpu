use std::time::Duration;

mod layer;
pub use layer::*;

use crate::{
    device::error::CudaError,
    runtime::{event::CudaEvent, ffi, stream::UnsafeCudaStream},
};

pub struct CudaInstant(pub(crate) CudaEvent);

unsafe impl Send for CudaInstant {}
unsafe impl Sync for CudaInstant {}

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

#[macro_export]
macro_rules! timed {
    ($level:expr, $msg:expr, $block:expr) => {{
        let start = CudaInstant::now().unwrap();
        let result = $block;
        tracing::event!($level, "{}: {:?}", $msg, start.elapsed().unwrap());
        result
    }};
}

#[macro_export]
macro_rules! timed_info {
    ($msg:expr, $block:expr) => {{
        let start = CudaInstant::now().unwrap();
        let result = $block;
        tracing::event!(
            tracing::Level::INFO,
            "{}: {:?}",
            $msg,
            start.elapsed().unwrap()
        );
        result
    }};
}

#[macro_export]
macro_rules! timed_debug {
    ($msg:expr, $block:expr) => {{
        let start = CudaInstant::now().unwrap();
        let result = $block;
        tracing::event!(
            tracing::Level::DEBUG,
            "{}, time: {:?}",
            $msg,
            start.elapsed().unwrap()
        );
        result
    }};
}

#[macro_export]
macro_rules! timed_warn {
    ($msg:expr, $block:expr) => {{
        let start = CudaInstant::now().unwrap();
        let result = $block;
        tracing::event!(
            tracing::Level::WARN,
            "{}, time: {:?}",
            $msg,
            start.elapsed().unwrap()
        );
        result
    }};
}

#[macro_export]
macro_rules! timed_error {
    ($msg:expr, $block:expr) => {{
        let start = CudaInstant::now().unwrap();
        let result = $block;
        tracing::event!(
            tracing::Level::ERROR,
            "{}, time: {:?}",
            $msg,
            start.elapsed().unwrap()
        );
        result
    }};
}
