use crate::mem::{copy_host_to_device, cuda_free, cuda_malloc};

/// Fixed-size device-side buffer.
#[derive(Debug)]
#[repr(C)]
pub struct DeviceBuffer<T: Copy> {
    buf: *mut T,
    len: usize,
    cap: usize,
}

unsafe impl<T: Send + Copy> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync + Copy> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let ptr = unsafe { cuda_malloc(capacity) }.unwrap();

        Self {
            buf: ptr,
            len: 0,
            cap: capacity,
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Copies all elements from `src` into `self`, using a cudaMemcpy.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths or if cudaMalloc
    /// returned an error.
    pub fn copy_from_slice(&mut self, src: &[T]) {
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

        unsafe { copy_host_to_device(self.buf, src.as_ptr(), src.len()) }.unwrap()
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
    pub fn extend_from_slice(&mut self, src: &[T]) {
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

        unsafe { copy_host_to_device(self.offset(), src.as_ptr(), src.len()) }.unwrap()
    }
}

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        unsafe { cuda_free(self.buf) }.unwrap()
    }
}
