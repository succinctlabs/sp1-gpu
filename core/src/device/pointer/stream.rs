use crate::cuda_runtime::stream::CudaStream;

use super::RawPointer;

pub struct DeviceStreamPointer<T> {
    ptr: *mut T,
    stream: CudaStream,
}

impl<T: Copy> RawPointer for DeviceStreamPointer<T> {
    type Data = T;

    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    fn free(&mut self) {
        unsafe { self.stream.cuda_free_async(self.ptr).unwrap() }
    }
}
