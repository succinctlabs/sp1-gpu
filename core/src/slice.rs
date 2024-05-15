use std::marker::PhantomData;

use crate::{error::CudaError, mem::copy_host_to_device};

/// Fixed-size device-side buffer.
#[derive(Debug)]
#[repr(C)]
pub struct DeviceSlice<T: Copy> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<[T]>,
}

unsafe impl<T: Sync + Copy> Sync for DeviceSlice<T> {}

impl<T: Copy> DeviceSlice<T> {
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Copies all elements from `src` into `self`, using a cudaMemcpy.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths.
    #[inline]
    pub fn copy_from_slice(&mut self, src: &[T]) -> Result<(), CudaError> {
        // The panic code path was put into a cold function to not bloat the
        // call site.
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn len_mismatch_fail(dst_len: usize, src_len: usize) -> ! {
            panic!(
                "source slice length ({}) does not match destination slice length ({})",
                src_len, dst_len,
            );
        }

        if self.len() != src.len() {
            len_mismatch_fail(self.len(), src.len());
        }

        unsafe { copy_host_to_device(self.ptr, src.as_ptr(), src.len()) }
    }
}
