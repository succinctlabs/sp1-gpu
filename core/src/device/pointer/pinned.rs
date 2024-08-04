use crate::device::{
    memory::{cuda_free_host, cuda_host_unregister, cuda_malloc_host},
    DefaultDeviceAllocator, DeviceAllocator, TryAllocError, DEFAULT_ALLOCATOR,
};

use super::{DefaultAllocatorPointer, Offset, RawPointer};

#[repr(transparent)]
pub struct CudaHostPointer<T>(*mut T);

pub struct CudaRegisteredPointer<T> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
}

impl<T: Copy> RawPointer for CudaHostPointer<T> {
    type Data = T;

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

impl<T: Copy> DefaultAllocatorPointer for CudaHostPointer<T> {
    type Allocator = DefaultDeviceAllocator;

    fn allocator(&self) -> &Self::Allocator {
        &DEFAULT_ALLOCATOR
    }
}

impl<T: Copy> RawPointer for CudaRegisteredPointer<T> {
    type Data = T;

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

impl<T: Copy> DeviceAllocator<CudaHostPointer<T>> for DefaultDeviceAllocator {
    unsafe fn try_alloc(&self, len: usize) -> Result<CudaHostPointer<T>, TryAllocError> {
        let ptr = cuda_malloc_host(len)?;

        Ok(CudaHostPointer(ptr))
    }
}

impl<T: Copy> Offset for CudaHostPointer<T> {
    unsafe fn add(&self, rhs: usize) -> Self {
        CudaHostPointer(self.0.add(rhs))
    }
}

impl<T: Copy> Offset for CudaRegisteredPointer<T> {
    unsafe fn add(&self, rhs: usize) -> Self {
        CudaRegisteredPointer {
            ptr: self.ptr.add(rhs),
            len: self.len,
            capacity: self.capacity,
        }
    }
}
