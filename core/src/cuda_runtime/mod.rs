use std::sync::{
    mpsc::{sync_channel, SyncSender},
    OnceLock,
};

use crate::device::error::CudaError;

static DEVICE_SENDER: OnceLock<SyncSender<TaskRef>> = OnceLock::new();

pub const DEFAULT_CAPACITY: usize = 100;

pub mod event;
pub(crate) mod ffi;
pub mod scope;
mod spawn;
pub mod stream;
pub mod task;

pub use scope::*;
pub use spawn::*;
use task::TaskRef;

pub fn sync_device() -> Result<(), CudaError> {
    unsafe { ffi::cuda_device_synchronize() }.into()
}

pub fn sync_default_stream() -> Result<(), CudaError> {
    unsafe { ffi::cuda_stream_synchronize(ffi::DEFAULT_STREAM) }.into()
}

fn current_device() -> &'static SyncSender<TaskRef> {
    DEVICE_SENDER.get_or_init(|| init_device(DEFAULT_CAPACITY))
}

fn init_device(capacity: usize) -> SyncSender<TaskRef> {
    let (sender, receiver) = sync_channel::<TaskRef>(capacity);

    std::thread::spawn(move || {
        for task in receiver.iter() {
            unsafe { task.execute() };
        }
    });

    sender
}
