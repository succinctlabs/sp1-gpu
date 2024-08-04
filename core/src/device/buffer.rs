use std::ops::{
    Deref, DerefMut, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
    RangeToInclusive,
};
use std::slice;

use crate::device::memory::{copy_device_to_host, copy_host_to_device};

use super::error::CudaError;
use super::memory::{CopyFrom, ToDevice, ToHost};
use super::{
    AllocError, CopyRawFrom, DefaultAllocatorPointer, DeviceAllocator, DevicePointer, RawPointer,
    TryAllocError, DEFAULT_ALLOCATOR,
};

/// Fixed-size device-side buffer.
#[derive(Debug)]
#[repr(C)]
pub struct Buffer<P: RawPointer> {
    buf: P,
    len: usize,
    cap: usize,
}

unsafe impl<P: RawPointer> Send for Buffer<P> {}
unsafe impl<P: RawPointer> Sync for Buffer<P> {}

pub type DeviceBuffer<T> = Buffer<DevicePointer<T>>;

impl<P: RawPointer> Buffer<P> {
    pub fn buf_ptr(&self) -> &P {
        &self.buf
    }

    pub fn try_with_capacity_in(
        capacity: usize,
        alloc: &impl DeviceAllocator<P>,
    ) -> Result<Self, TryAllocError> {
        let buf = unsafe { alloc.try_alloc(capacity)? };

        Ok(Self {
            buf,
            len: 0,
            cap: capacity,
        })
    }

    pub fn with_capacity_in(
        capacity: usize,
        alloc: &impl DeviceAllocator<P>,
    ) -> Result<Self, AllocError> {
        let buf = unsafe { alloc.alloc(capacity)? };

        Ok(Self {
            buf,
            len: 0,
            cap: capacity,
        })
    }

    pub fn allocator(&self) -> &P::Allocator
    where
        P: DefaultAllocatorPointer,
    {
        self.buf.allocator()
    }

    /// Returns a new buffer from a pointer, length, and capacity.
    ///
    /// # Safety
    ///
    /// The pointer must be valid, it must have allocated memory in the size of
    /// capacity * size_of<T>, and the first `len` elements of the buffer must be initialized or
    /// about to be initialized in a foreign CUDA call.
    pub const unsafe fn from_raw_parts(ptr: P, length: usize, capacity: usize) -> Self {
        Self {
            buf: ptr,
            len: length,
            cap: capacity,
        }
    }

    /// # Safety
    ///
    /// TODO
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        self.len = new_len;
    }

    /// # Safety
    ///
    /// TODO
    #[inline]
    pub unsafe fn set_max_len(&mut self) {
        self.len = self.cap;
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    // pub fn as_slice(&self) -> &DeviceSlice<P> {
    //     &self[..]
    // }

    // pub fn as_slice_mut(&mut self) -> &mut DeviceSlice<P> {
    //     &mut self[..]
    // }

    pub fn as_ptr(&self) -> *const P::Data {
        self.buf.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut P::Data {
        self.buf.as_mut_ptr()
    }

    /// Calculates the offset to the current element.
    #[inline]
    unsafe fn offset(&self) -> *mut P::Data {
        (self.buf.as_ptr() as *mut P::Data).add(self.len)
    }
}

impl<P: RawPointer> CopyFrom<[P::Data]> for Buffer<P>
where
    P: CopyRawFrom<*const P::Data>,
{
    fn copy_from(&mut self, src: &[P::Data], len: usize) -> Result<(), CudaError> {
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

        unsafe { self.buf.copy_raw_from(&src.as_ptr(), len) }
    }
}

impl<T: Copy> DeviceBuffer<T> {
    pub fn with_capacity(capacity: usize) -> Result<Self, CudaError> {
        Self::with_capacity_in(capacity, &DEFAULT_ALLOCATOR).map_err(|e| e.0)
    }

    /// Copies all elements from `src` into `self`, using a cudaMemcpy.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths or if cudaMalloc
    /// returned an error.
    pub fn copy_from_host(&mut self, src: &[T]) {
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

        unsafe { copy_host_to_device(self.buf.as_mut_ptr(), src.as_ptr(), src.len()) }.unwrap()
    }

    pub fn copy_to_host(&self, dst: &mut [T]) {
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

        unsafe { copy_device_to_host(dst.as_mut_ptr(), self.buf.as_ptr(), dst.len()) }.unwrap()
    }

    /// Appends all the elements from `src` into `self`, using a cudaMemcpy.
    ///
    /// # Panics
    ///
    /// This function will panic if the resulting length will extend the buffer's capacity or if
    /// cudaMalloc returned an error.
    pub fn extend_from_host_slice(&mut self, src: &[T]) {
        // The panic code path was put into a cold function to not bloat the
        // call site.
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn capacity_fail(dst_len: usize, src_len: usize, cap: usize) -> ! {
            panic!(
                "source slice length ({}) too long for buffer of length ({}) and capacity ({})",
                src_len, dst_len, cap
            );
        }

        if self.len() + src.len() > self.cap {
            capacity_fail(self.len(), src.len(), self.cap);
        }

        unsafe { copy_host_to_device(self.offset(), src.as_ptr(), src.len()) }.unwrap();

        // Extend the length of the buffer to include the new elements.
        self.len += src.len();
    }
}

impl<P: RawPointer> Drop for Buffer<P> {
    fn drop(&mut self) {
        self.buf.free()
    }
}

// macro_rules! impl_index {
//     ($($t:ty)*) => {
//         $(
//             impl<P: RawPointer> Index<$t> for Buffer<P>
//             {
//                 type Output = DeviceSlice<P>;

//                 fn index(&self, index: $t) -> &DeviceSlice<P> {
//                     unsafe {
//                         DeviceSlice::from_slice(
//                          slice::from_raw_parts(self.buf.as_ptr(), self.len).index(index)
//                     )
//                   }
//                 }
//             }

//             impl<P: RawPointer> IndexMut<$t> for Buffer<P>
//             {
//                 fn index_mut(&mut self, index: $t) -> &mut DeviceSlice<P> {
//                     unsafe {
//                         DeviceSlice::from_slice_mut(
//                             slice::from_raw_parts_mut(self.buf.as_mut_ptr(), self.len).index_mut(index)
//                         )
//                     }
//                 }
//             }
//         )*
//     }
// }

// impl_index! {
//     Range<usize>
//     RangeFull
//     RangeFrom<usize>
//     RangeInclusive<usize>
//     RangeTo<usize>
//     RangeToInclusive<usize>
// }

impl<T: Copy> ToDevice for Vec<T> {
    type DeviceType = DeviceBuffer<T>;

    fn to_device(&self) -> Result<Self::DeviceType, CudaError> {
        let mut buffer = DeviceBuffer::with_capacity(self.len())?;
        buffer.extend_from_host_slice(self);
        Ok(buffer)
    }
}

impl<T: Copy> ToHost for DeviceBuffer<T> {
    type HostType = Vec<T>;

    fn to_host(&self) -> Vec<T> {
        let mut host = Vec::with_capacity(self.len);
        unsafe {
            host.set_len(self.len);
        }
        self.copy_to_host(&mut host);
        host
    }
}

// impl<P: RawPointer> Deref for Buffer<P> {
//     type Target = DeviceSlice<P>;

//     #[inline]
//     fn deref(&self) -> &Self::Target {
//         &self[..]
//     }
// }

// impl<T: Copy> DerefMut for DeviceBuffer<T> {
//     #[inline]
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self[..]
//     }
// }

#[cfg(test)]
mod tests {
    use rand::{
        distributions::{Distribution, Standard},
        thread_rng, Rng,
    };
    use std::fmt::Debug;

    use super::DeviceBuffer;

    fn make_test_buffer_init_and_copy<T>(rng: &mut impl Rng, len: usize)
    where
        T: Debug + Copy + Default + Eq,
        Standard: Distribution<T>,
    {
        let mut buffer = DeviceBuffer::<T>::with_capacity(len).unwrap();
        assert_eq!(buffer.len(), 0);

        let values = (0..len).map(|_| rng.gen()).collect::<Vec<_>>();
        buffer.extend_from_host_slice(&values);
        assert_eq!(buffer.len(), values.len());

        let mut values_back = vec![T::default(); len];
        buffer.copy_to_host(&mut values_back);

        for (val, exp) in values_back.into_iter().zip(values) {
            assert_eq!(val, exp);
        }
    }

    // fn make_test_buffer_slice_index<T>(rng: &mut impl Rng, len: usize, slice_range: Range<usize>)
    // where
    //     T: Debug + Copy + Default + Eq,
    //     Standard: Distribution<T>,
    // {
    //     let mut buffer = DeviceBuffer::<T>::with_capacity(len).unwrap();
    //     assert_eq!(buffer.len(), 0);

    //     // Initialize the buffer to zero.
    //     buffer.extend_from_host_slice(&vec![T::default(); len]);
    //     assert_eq!(buffer.len(), len);

    //     let new_values = slice_range.clone().map(|_| rng.gen()).collect::<Vec<_>>();
    //     let device_slice = &mut buffer[slice_range.clone()];
    //     assert_eq!(device_slice.len(), slice_range.len());

    //     device_slice.copy_from(&new_values);

    //     let mut new_values_back = vec![T::default(); len];
    //     device_slice.copy_into_host(&mut new_values_back[0..slice_range.len()]);

    //     for (val, exp) in new_values_back.into_iter().zip(new_values) {
    //         assert_eq!(val, exp);
    //     }
    // }

    #[test]
    fn test_buffer_init_and_copy() {
        let len = 10000;

        let mut rng = thread_rng();
        make_test_buffer_init_and_copy::<u32>(&mut rng, len);
        make_test_buffer_init_and_copy::<u64>(&mut rng, len);
    }

    // #[test]
    // fn test_buffer_slice_index() {
    //     let len = 10000;
    //     let range = 34..900;

    //     let mut rng = thread_rng();
    //     make_test_buffer_slice_index::<u32>(&mut rng, len, range.clone());
    //     make_test_buffer_slice_index::<u64>(&mut rng, len, range);
    // }
}
