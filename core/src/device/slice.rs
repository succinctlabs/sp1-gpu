use core::slice;
use std::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use super::RawPointer;

#[derive(Debug)]
#[repr(transparent)]
pub struct DeviceSlice<P: RawPointer>([P::Data]);

impl<P: RawPointer> DeviceSlice<P> {
    #[inline]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const P::Data {
        self.0.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut P::Data {
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
    pub(crate) unsafe fn from_slice(src: &[P::Data]) -> &Self {
        &*(src as *const [P::Data] as *const Self)
    }

    #[inline(always)]
    pub(crate) unsafe fn from_slice_mut(src: &mut [P::Data]) -> &mut Self {
        &mut *(src as *mut [P::Data] as *mut Self)
    }

    /// # Safety
    pub unsafe fn from_raw_parts<'a>(data: *const P::Data, len: usize) -> &'a Self {
        Self::from_slice(slice::from_raw_parts(data, len))
    }

    /// # Safety
    pub unsafe fn from_raw_parts_mut<'a>(data: *mut P::Data, len: usize) -> &'a mut Self {
        Self::from_slice_mut(slice::from_raw_parts_mut(data, len))
    }

    // /// Copies all elements from `src` into `self`, using a cudaMemcpy.
    // ///
    // /// The length of `src` must be the same as `self`.
    // ///
    // /// # Panics
    // ///
    // /// This function will panic if the two slices have different lengths or if cudaMalloc
    // /// returned an error.
    // pub fn copy_from_host(&mut self, src: &[P::Data])
    // where
    //     P::Data: Copy,
    // {
    //     // The panic code path was put into a cold function to not bloat the
    //     // call site.
    //     #[inline(never)]
    //     #[cold]
    //     #[track_caller]
    //     fn len_mismatch_fail(dst_len: usize, src_len: usize) -> ! {
    //         panic!(
    //             "source slice length ({}) does not match destination slice length ({})",
    //             src_len, dst_len,
    //         );
    //     }

    //     if self.len() != src.len() {
    //         len_mismatch_fail(self.len(), src.len());
    //     }

    //     unsafe { copy_host_to_device(self.0.as_mut_ptr(), src.as_ptr(), src.len()) }.unwrap()
    // }

    // /// Copies all elements from `src` into `self`, using a cudaMemcpy.
    // ///
    // /// The length of `src` must be the same as `self`.
    // ///
    // /// # Panics
    // ///
    // /// This function will panic if the two slices have different lengths or if cudaMalloc
    // /// returned an error.
    // pub fn copy_into_host(&self, dst: &mut [P::Data])
    // where
    //     P::Data: Copy,
    // {
    //     // The panic code path was put into a cold function to not bloat the
    //     // call site.
    //     #[inline(never)]
    //     #[cold]
    //     #[track_caller]
    //     fn len_mismatch_fail(dst_len: usize, src_len: usize) -> ! {
    //         panic!(
    //             "source slice length ({}) does not match destination slice length ({})",
    //             src_len, dst_len,
    //         );
    //     }

    //     if self.len() != dst.len() {
    //         len_mismatch_fail(self.len(), dst.len());
    //     }

    //     unsafe { copy_device_to_host(dst.as_mut_ptr(), self.0.as_ptr(), dst.len()) }.unwrap()
    // }

    // /// Copies all elements from `src` into `self`, using a cudaMemcpy.
    // ///
    // /// The length of `src` must be the same as `self`.
    // ///
    // /// # Panics
    // ///
    // /// This function will panic if the two slices have different lengths or if cudaMalloc
    // /// returned an error.
    // pub fn copy_from_device(&mut self, src: &DeviceSlice<T>) -> Result<(), CudaError>
    // where
    //     T: Copy,
    // {
    //     // The panic code path was put into a cold function to not bloat the
    //     // call site.
    //     #[inline(never)]
    //     #[cold]
    //     #[track_caller]
    //     fn len_mismatch_fail(dst_len: usize, src_len: usize) -> ! {
    //         panic!(
    //             "source slice length ({}) does not match destination slice length ({})",
    //             src_len, dst_len,
    //         );
    //     }

    //     if self.len() != src.len() {
    //         len_mismatch_fail(self.len(), src.len());
    //     }

    //     unsafe { copy_device_to_device(self.as_mut_ptr(), src.0.as_ptr(), src.len()) }
    // }
}

macro_rules! impl_index {
    ($($t:ty)*) => {
        $(
            impl<P: RawPointer> Index<$t> for DeviceSlice<P>
            {
                type Output = DeviceSlice<P>;

                fn index(&self, index: $t) -> &Self {
                    unsafe { DeviceSlice::from_slice(self.0.index(index)) }
                }
            }

            impl<P: RawPointer> IndexMut<$t> for DeviceSlice<P>
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

// impl<P: RawPointer> ToHost for DeviceSlice<P>
// where
//     P: CopyRawTo<*mut P::Data>,
// {
//     type HostType = Vec<P::Data>;

//     fn to_host(&self) -> Vec<P::Data> {
//         let mut host = Vec::with_capacity(self.len());
//         unsafe {
//             host.set_len(self.len());
//         }
//         self.copy_into_host(&mut host);
//         host
//     }
// }
