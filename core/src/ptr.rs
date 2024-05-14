use std::{ffi::c_void, marker::PhantomData};

// See https://github.com/Rust-GPU/Rust-CUDA/blob/master/crates/cust/src/memory/pointer.rs#L26

/// A pointer to device memory.
///
/// `DevicePointer` cannot be dereferenced by the CPU, as it is a pointer to a memory allocation in
/// the device. It can be safely copied to the device (eg. as part of a kernel launch) and either
/// unwrapped or transmuted to an appropriate pointer.
///
/// `DevicePointer` is guaranteed to have an equivalent internal representation to a raw pointer.
/// Thus, it can be safely reinterpreted or transmuted to `*mut T`. It is safe to pass a
/// `DevicePointer` through an FFI boundary to C code expecting a `*mut T`, so long as the code on
/// the other side of that boundary does not attempt to dereference the pointer on the CPU. It is
/// thus possible to pass a `DevicePointer` to a CUDA kernel written in C.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct DevicePointer<T: ?Sized + Copy> {
    ptr: *mut c_void,
    marker: PhantomData<*mut T>,
}

impl<T: ?Sized + Copy> DevicePointer<T> {
    /// Ctreates a device pointer from a raw pointer.
    ///
    /// # Safety
    /// Raw pointer must be a CUDA pointer.
    #[inline]
    pub const unsafe fn from_raw(ptr: *mut c_void) -> Self {
        DevicePointer {
            ptr,
            marker: PhantomData,
        }
    }

    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr as *mut T
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn as_raw_ptr(&self) -> *const c_void {
        self.ptr as *const c_void
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn as_mut_raw_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Returns true if the pointer is null.
    pub fn is_null(self) -> bool {
        self.ptr.is_null()
    }
}
