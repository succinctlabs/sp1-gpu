use crate::device::memory::{cuda_free_host, cuda_host_unregister};

use super::RawPointer;

#[repr(transparent)]
pub struct CudaHostPointer<T>(*mut T);

pub struct CudaRegisteredPointer<T> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
}

impl<T: Copy> RawPointer<T> for CudaHostPointer<T> {
    fn as_ptr(&self) -> *const T {
        self.0
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.0
    }

    fn free(&mut self) {
        unsafe { cuda_free_host(self.0).unwrap() }
    }
}

impl<T: Copy> RawPointer<T> for CudaRegisteredPointer<T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    fn free(&mut self) {
        // Unregister the pointer.
        unsafe { cuda_host_unregister(self.ptr).unwrap() }
        // Free the memory by creating a vector.
        let vec = unsafe { Vec::from_raw_parts(self.ptr, self.len, self.capacity) };
        // Drop the vector. This call is not necessary since the vector will be dropped in the
        // end of the scope, but we keep it here for clarity.
        drop(vec);
    }
}
