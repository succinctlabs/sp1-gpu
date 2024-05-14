use std::{ffi::c_void, mem, ptr};

use crate::{error::CudaError, ffi, ptr::DevicePointer};

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
pub unsafe fn cuda_free<T: Copy>(ptr: &DevicePointer<T>) -> Result<(), CudaError> {
    ffi::cuda_free(ptr.as_raw_ptr()).into()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_cuda_malloc() {
        let len = 1 << 4;
        let ptr: DevicePointer<u32> = unsafe { cuda_malloc(len) }.unwrap();
        unsafe { cuda_free(&ptr) }.unwrap();
    }
}
