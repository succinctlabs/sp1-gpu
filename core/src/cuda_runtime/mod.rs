use std::sync::{
    mpsc::{sync_channel, SyncSender},
    OnceLock,
};

use crate::device::{error::CudaError, memory::GlobalDeviceAllocator};

static DEVICE_SENDER: OnceLock<SyncSender<TaskRef>> = OnceLock::new();

pub const GLOBAL_DEVICE_ALLOCATOR: GlobalDeviceAllocator = GlobalDeviceAllocator;

pub const DEFAULT_CAPACITY: usize = 100;

pub mod event;
pub(crate) mod ffi;
pub mod scope;
pub mod spawn;
pub mod stream;
pub mod task;

use moongate_bloc::{alloc::Allocator, bump::Bump};
pub use scope::*;
pub use spawn::*;

use stream::CudaStream;
use task::TaskRef;

pub trait CudaSync {
    fn stream(&self) -> &CudaStream;
}

impl<A: Allocator + CudaSync> CudaSync for Bump<A> {
    fn stream(&self) -> &CudaStream {
        self.pool_allocator().stream()
    }
}

pub trait DeviceAllocator: Allocator + CudaSync + Send + Clone {}

impl<A> DeviceAllocator for A where A: Allocator + CudaSync + Send + Clone {}

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

#[cfg(test)]
mod tests {
    use moongate_bloc::bump::Bump;

    use super::stream::CudaStream;

    #[test]
    fn test_bump_allocations() {
        let bump = Bump::with_capacity_in(100, CudaStream::default());

        // let buffer = DeviceBuffer::<u32, _>::with_capacity_in(95, &bump);
    }
}
