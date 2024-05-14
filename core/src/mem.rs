use std::{ffi::c_void, marker::PhantomData, mem, ptr};

use crate::{error::CudaError, ffi};

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

    /// Returns true if the pointer is null.
    pub fn is_null(self) -> bool {
        self.ptr.is_null()
    }
}

/// A Rust interface for cudaMalloc.
///
/// # Safety
/// The pointer will not be dropped unless a `cudaFree` call is initiated.
pub unsafe fn cuda_malloc<T: Copy>(len: usize) -> Result<DevicePointer<T>, CudaError> {
    let mut ptr: *mut c_void = ptr::null_mut();
    Result::<(), CudaError>::from(ffi::cuda_malloc(
        &mut ptr as *mut *mut c_void,
        len * mem::size_of::<T>(),
    ))?;

    Ok(DevicePointer::from_raw(ptr))
}

/// A Rust interface for cudaFree.
///
/// # Safety
pub unsafe fn cuda_free<T: Copy>(ptr: &mut DevicePointer<T>) -> Result<(), CudaError> {
    ffi::cuda_free(ptr.ptr).into()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_cuda_malloc() {
        let len = 1 << 40;
        let mut ptr: DevicePointer<u32> = unsafe { cuda_malloc(len) }.unwrap();
        unsafe { cuda_free(&mut ptr) }.unwrap();
    }
}
