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

impl<'a, T: CudaSync + ?Sized> CudaSync for &'a T {
    fn stream(&self) -> &CudaStream {
        (**self).stream()
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

    use crate::device::DeviceBuffer;

    use super::stream::CudaStream;

    use crate::device::memory::ToHost;

    #[test]
    fn test_bump_set_values() {
        let bump = Bump::<CudaStream>::default();
        let mut buffer = DeviceBuffer::<u8, _>::with_capacity_in(100, &bump).unwrap();
        unsafe {
            buffer.set_len(100);
        }
        buffer.set(121).unwrap();
        let host = buffer.to_host();
        for val in host {
            assert_eq!(val, 121);
        }
    }

    #[test]
    fn test_bump_allocation_limit() {
        let bump = Bump::<CudaStream>::default();
        bump.set_allocation_limit(Some(50));
        let res = DeviceBuffer::<u8, _>::with_capacity_in(100, &bump);
        assert!(res.is_err())
    }
}
