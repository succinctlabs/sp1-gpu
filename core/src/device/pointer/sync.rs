use crate::device::memory::cuda_free;

use super::RawPointer;

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct DevicePointer<T>(*mut T);

impl<T> RawPointer<T> for DevicePointer<T> {
    fn as_ptr(&self) -> *const T {
        self.0
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.0
    }

    fn free(self) {
        unsafe { cuda_free(self.0) }.unwrap()
    }
}
