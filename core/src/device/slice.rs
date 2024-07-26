use crate::device::memory::{copy_device_to_device, copy_device_to_host, copy_host_to_device};
use core::slice;
use std::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use super::{error::CudaError, memory::ToHost};

#[derive(Debug)]
#[repr(transparent)]
pub struct DeviceSlice<T>([T]);

impl<T> DeviceSlice<T> {
    #[inline]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }

    #[inline]
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut Self, &mut Self) {
        let (left, right) = self.0.split_at_mut(mid);
        unsafe { (Self::from_slice_mut(left), Self::from_slice_mut(right)) }
    }

    #[inline]
    pub fn split_at(&self, mid: usize) -> (&Self, &Self) {
        let (left, right) = self.0.split_at(mid);
        unsafe { (Self::from_slice(left), Self::from_slice(right)) }
    }

    #[inline(always)]
    pub(crate) unsafe fn from_slice(src: &[T]) -> &Self {
        &*(src as *const [T] as *const Self)
    }

    #[inline(always)]
    pub(crate) unsafe fn from_slice_mut(src: &mut [T]) -> &mut Self {
        &mut *(src as *mut [T] as *mut Self)
    }

    /// # Safety
    pub unsafe fn from_raw_parts<'a>(data: *const T, len: usize) -> &'a Self {
        Self::from_slice(slice::from_raw_parts(data, len))
    }

    /// # Safety
    pub unsafe fn from_raw_parts_mut<'a>(data: *mut T, len: usize) -> &'a mut Self {
        Self::from_slice_mut(slice::from_raw_parts_mut(data, len))
    }

    /// Copies all elements from `src` into `self`, using a cudaMemcpy.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths or if cudaMalloc
    /// returned an error.
    pub fn copy_from_host(&mut self, src: &[T])
    where
        T: Copy,
    {
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

        unsafe { copy_host_to_device(self.0.as_mut_ptr(), src.as_ptr(), src.len()) }.unwrap()
    }

    /// Copies all elements from `src` into `self`, using a cudaMemcpy.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths or if cudaMalloc
    /// returned an error.
    pub fn copy_into_host(&self, dst: &mut [T])
    where
        T: Copy,
    {
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

        if self.len() != dst.len() {
            len_mismatch_fail(self.len(), dst.len());
        }

        unsafe { copy_device_to_host(dst.as_mut_ptr(), self.0.as_ptr(), dst.len()) }.unwrap()
    }

    /// Copies all elements from `src` into `self`, using a cudaMemcpy.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths or if cudaMalloc
    /// returned an error.
    pub fn copy_from_device(&mut self, src: &DeviceSlice<T>) -> Result<(), CudaError>
    where
        T: Copy,
    {
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

        unsafe { copy_device_to_device(self.as_mut_ptr(), src.0.as_ptr(), src.len()) }
    }
}

macro_rules! impl_index {
    ($($t:ty)*) => {
        $(
            impl<T : Copy> Index<$t> for DeviceSlice<T>
            {
                type Output = DeviceSlice<T>;

                fn index(&self, index: $t) -> &Self {
                    unsafe { DeviceSlice::from_slice(self.0.index(index)) }
                }
            }

            impl<T : Copy> IndexMut<$t> for DeviceSlice<T>
            {
                fn index_mut(&mut self, index: $t) -> &mut Self {
                    unsafe { DeviceSlice::from_slice_mut( self.0.index_mut(index)) }
                }
            }
        )*
    }
}

impl_index! {
    Range<usize>
    RangeFull
    RangeFrom<usize>
    RangeInclusive<usize>
    RangeTo<usize>
    RangeToInclusive<usize>
}

impl<T: Copy> ToHost for DeviceSlice<T> {
    type HostType = Vec<T>;

    fn to_host(&self) -> Vec<T> {
        let mut host = Vec::with_capacity(self.len());
        unsafe {
            host.set_len(self.len());
        }
        self.copy_into_host(&mut host);
        host
    }
}
