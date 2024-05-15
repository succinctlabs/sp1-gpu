use std::ffi::c_void;

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
pub struct DevicePointer<T: ?Sized + Copy>(*mut T);

impl<T: ?Sized + Copy> DevicePointer<T> {
    /// Ctreates a device pointer from a raw pointer.
    ///
    /// # Safety
    /// Raw pointer must be a CUDA pointer.
    #[inline]
    pub const unsafe fn from_raw(ptr: *mut c_void) -> Self {
        DevicePointer(ptr as *mut T)
    }

    #[inline]
    pub const fn as_ptr(&self) -> *const T {
        self.0 as *const T
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn as_raw_ptr(&self) -> *const c_void {
        self.0 as *const c_void
    }

    #[inline]
    #[allow(dead_code)]
    pub(crate) const fn as_mut_raw_ptr(&self) -> *mut c_void {
        self.0 as *mut c_void
    }

    /// Returns true if the pointer is null.
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}

/// A pointer to unified memory.
///
/// `UnifiedPointer` can be safely dereferenced by the CPU, as the memory allocation it points to is
/// shared between the CPU and the GPU. It can also be safely copied to the device (eg. as part of
/// a kernel launch).
///
/// `UnifiedPointer` is guaranteed to have an equivalent internal representation to a raw pointer.
/// Thus, it can be safely reinterpreted or transmuted to `*mut T`. It is also safe to pass a
/// `UnifiedPointer` through an FFI boundary to C code expecting a `*mut T`. It is
/// thus possible to pass a `UnifiedPointer` to a CUDA kernel written in C.
#[repr(transparent)]
pub struct UnifiedPointer<T: ?Sized>(*mut T);
