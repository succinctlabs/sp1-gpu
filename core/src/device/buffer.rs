use std::{
    mem,
    ops::{
        Deref, DerefMut, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
        RangeToInclusive,
    },
    slice,
    time::Duration,
};

use p3_field::{ExtensionField, Field, PrimeField32};

use crate::{
    cuda_runtime::stream::{AllocTimeoutError, CudaStream},
    device::slice::DeviceSlice,
};

use super::{
    error::CudaError,
    memory::{ToDevice, ToHost},
};

/// Fixed-size device-side buffer.
#[derive(Debug)]
#[repr(C)]
pub struct DeviceBuffer<T: Copy> {
    buf: *mut T,
    len: usize,
    cap: usize,
    stream: CudaStream,
}

unsafe impl<T: Copy> Send for DeviceBuffer<T> {}
unsafe impl<T: Copy> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocate a new buffer on the device.
    ///
    /// The function will return an error if there is not enough memory available, or if any other
    /// device error occurs.
    pub fn with_capacity(capacity: usize) -> Result<Self, CudaError> {
        Self::with_capacity_in(capacity, &CudaStream::default())
    }

    /// Creates a buffer with a null pointer and zero capacity.
    pub fn null() -> Self {
        Self { buf: std::ptr::null_mut(), len: 0, cap: 0, stream: CudaStream::default() }
    }

    /// Allocate a new buffer on the device.
    ///
    /// The function will return an error if there is not enough memory available, or if any other
    /// device error occurs.
    pub fn try_with_capacity_in(capacity: usize, stream: &CudaStream) -> Result<Self, CudaError> {
        let ptr = unsafe { stream.try_alloc(capacity) }?;

        Ok(Self { buf: ptr, len: 0, cap: capacity, stream: stream.clone() })
    }

    /// Allocate a new buffer on the device.
    ///
    /// The function will block until enough memory is available. The function will return an error
    /// if another device error occurs.
    pub fn with_capacity_in(capacity: usize, stream: &CudaStream) -> Result<Self, CudaError> {
        let ptr = unsafe { stream.alloc(capacity) }?;

        Ok(Self { buf: ptr, len: 0, cap: capacity, stream: stream.clone() })
    }

    /// Allocate a new buffer on the device.
    ///
    /// The function will block until enough memory is available or the timeout is reached. The
    /// function will return an error if another device error occurs.
    pub fn with_capacity_in_timeout(
        capacity: usize,
        stream: CudaStream,
        timeout: Duration,
    ) -> Result<Self, AllocTimeoutError> {
        let ptr = unsafe { stream.alloc_timeout(capacity, timeout) }?;

        Ok(Self { buf: ptr, len: 0, cap: capacity, stream })
    }

    /// Returns a new buffer from a pointer, length, and capacity.
    ///
    /// # Safety
    ///
    /// The pointer must be valid, it must have allocated memory in the size of
    /// capacity * size_of<T>, and the first `len` elements of the buffer must be initialized or
    /// about to be initialized in a foreign CUDA call.
    pub unsafe fn from_raw_parts(
        ptr: *mut T,
        length: usize,
        capacity: usize,
        stream: CudaStream,
    ) -> Self {
        Self { buf: ptr, len: length, cap: capacity, stream }
    }

    #[inline]
    pub const fn stream(&self) -> &CudaStream {
        &self.stream
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

    #[inline]
    pub fn as_slice(&self) -> &DeviceSlice<T> {
        &self[..]
    }

    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut DeviceSlice<T> {
        &mut self[..]
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.buf
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.buf
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

        unsafe {
            self.stream
                .cuda_memcpy_host_to_device_async(self.buf, src.as_ptr(), src.len())
                .unwrap();
        }
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

        unsafe {
            self.stream
                .cuda_memcpy_device_to_host_async(dst.as_mut_ptr(), self.buf, dst.len())
                .unwrap();
        }
    }

    /// Calculates the offset to the current element.
    #[inline]
    unsafe fn offset(&self) -> *mut T {
        self.buf.add(self.len)
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

        unsafe {
            self.stream
                .cuda_memcpy_host_to_device_async(self.offset(), src.as_ptr(), src.len())
                .unwrap();
        }

        // Extend the length of the buffer to include the new elements.
        self.len += src.len();
    }

    /// # Safety
    ///
    /// A device slice of type `T` should be castable to device slice of type `B`.
    pub unsafe fn flatten_to_base<B>(self) -> DeviceBuffer<B>
    where
        B: PrimeField32,
        T: ExtensionField<B>,
    {
        // Cast the device pointer to the base type.
        let buff = self.buf as *mut B;
        // Clone the stream.
        let stream = self.stream.clone();

        // The new length/capacity are the product of the old length/capacity and the extension
        // degree.
        let len = self.len * T::D;
        let cap = self.cap * T::D;

        // Prevent the buffer from being dropped.
        mem::forget(self);

        DeviceBuffer::from_raw_parts(buff, len, cap, stream)
    }

    /// # Safety
    ///
    /// A device slice of type `T` should be castable to device slice of type `B`.
    pub unsafe fn as_extension_buffer<E>(self) -> DeviceBuffer<E>
    where
        T: Field,
        E: ExtensionField<T>,
    {
        // Cast the device pointer to the extension type.
        let buff = self.buf as *mut E;
        // Clone the stream.
        let stream = self.stream.clone();

        // The legth and capacity must be divisible by the degree.
        assert!(self.len % E::D == 0);
        assert!(self.cap % E::D == 0);
        let len = self.len / E::D;
        let cap = self.cap / E::D;

        // Prevent the buffer from being dropped.
        mem::forget(self);

        DeviceBuffer::from_raw_parts(buff, len, cap, stream)
    }
}

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        unsafe { self.stream.free_async(self.buf).unwrap() }
    }
}

macro_rules! impl_index {
    ($($t:ty)*) => {
        $(
            impl<T : Copy> Index<$t> for DeviceBuffer<T>
            {
                type Output = DeviceSlice<T>;

                fn index(&self, index: $t) -> &DeviceSlice<T> {
                    unsafe {
                        DeviceSlice::from_slice(
                         slice::from_raw_parts(self.buf, self.len).index(index)
                    )
                  }
                }
            }

            impl<T : Copy> IndexMut<$t> for DeviceBuffer<T>
            {
                fn index_mut(&mut self, index: $t) -> &mut DeviceSlice<T> {
                    unsafe {
                        DeviceSlice::from_slice_mut(
                            slice::from_raw_parts_mut(self.buf, self.len).index_mut(index)
                        )
                    }
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

impl<T: Copy> ToDevice for Vec<T> {
    type DeviceType = DeviceBuffer<T>;

    fn to_device_async(&self, stream: &CudaStream) -> Result<Self::DeviceType, CudaError> {
        let mut buffer = DeviceBuffer::with_capacity_in(self.len(), stream)?;
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

impl<T: Copy> Deref for DeviceBuffer<T> {
    type Target = DeviceSlice<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self[..]
    }
}

impl<T: Copy> DerefMut for DeviceBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self[..]
    }
}

#[cfg(test)]
mod tests {
    use rand::{
        distributions::{Distribution, Standard},
        thread_rng, Rng,
    };
    use std::{fmt::Debug, ops::Range};

    use crate::cuda_runtime::stream::CudaStream;

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

    fn make_test_buffer_slice_index<T>(rng: &mut impl Rng, len: usize, slice_range: Range<usize>)
    where
        T: Debug + Copy + Default + Eq,
        Standard: Distribution<T>,
    {
        let mut buffer = DeviceBuffer::<T>::with_capacity(len).unwrap();
        assert_eq!(buffer.len(), 0);

        // Initialize the buffer to zero.
        buffer.extend_from_host_slice(&vec![T::default(); len]);
        assert_eq!(buffer.len(), len);

        let new_values = slice_range.clone().map(|_| rng.gen()).collect::<Vec<_>>();
        let device_slice = &mut buffer[slice_range.clone()];
        assert_eq!(device_slice.len(), slice_range.len());

        device_slice.copy_from_host(&new_values, &CudaStream::default());

        let mut new_values_back = vec![T::default(); len];
        device_slice
            .copy_into_host(&mut new_values_back[0..slice_range.len()], &CudaStream::default());

        for (val, exp) in new_values_back.into_iter().zip(new_values) {
            assert_eq!(val, exp);
        }
    }

    #[test]
    fn test_buffer_init_and_copy() {
        let len = 10000;

        let mut rng = thread_rng();
        make_test_buffer_init_and_copy::<u32>(&mut rng, len);
        make_test_buffer_init_and_copy::<u64>(&mut rng, len);
    }

    #[test]
    fn test_buffer_slice_index() {
        let len = 10000;
        let range = 34..900;

        let mut rng = thread_rng();
        make_test_buffer_slice_index::<u32>(&mut rng, len, range.clone());
        make_test_buffer_slice_index::<u64>(&mut rng, len, range);
    }
}
