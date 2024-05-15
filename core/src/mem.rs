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
/// The caller must guarantee that after this call no data will point to the value of the pointer.
pub unsafe fn cuda_free<T: Copy>(ptr: &DevicePointer<T>) -> Result<(), CudaError> {
    ffi::cuda_free(ptr.as_raw_ptr()).into()
}

/// A Rust interface for cudaMemcpy from host to device.
///
/// # Safety
/// The caller must guarantee that the memory copied from/into size is valid.
pub unsafe fn copy_host_to_device<T: Copy>(
    dst: &mut DevicePointer<T>,
    src: &[T],
    len: usize,
) -> Result<(), CudaError> {
    ffi::cuda_mem_copy_host_to_device(
        dst.as_mut_ptr() as *mut c_void,
        src.as_ptr() as *const c_void,
        len * mem::size_of::<T>(),
    )
    .into()
}

/// A Rust interface for cudaMemcpy from device to host.
///
/// # Safety
/// The caller must guarantee that the memory copied from/into size is valid.
pub unsafe fn copy_device_to_host<T: Copy>(
    dst: &mut [T],
    src: &DevicePointer<T>,
    len: usize,
) -> Result<(), CudaError> {
    ffi::cuda_mem_copy_device_to_host(
        dst.as_mut_ptr() as *mut c_void,
        src.as_ptr() as *const c_void,
        len * mem::size_of::<T>(),
    )
    .into()
}

/// A Rust interface for cudaMemcpy.
///
/// # Safety
/// The caller must guarantee that the memory copied from/into size is valid.
pub unsafe fn copy_device_to_device<T: Copy>(
    dst: &mut [T],
    src: &DevicePointer<T>,
    len: usize,
) -> Result<(), CudaError> {
    ffi::cuda_mem_copy_device_to_device(
        dst.as_mut_ptr() as *mut c_void,
        src.as_ptr() as *const c_void,
        len * mem::size_of::<T>(),
    )
    .into()
}

#[cfg(test)]
mod tests {

    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn test_cuda_malloc() {
        let len = 1 << 4;
        let ptr: DevicePointer<u32> = unsafe { cuda_malloc(len) }.unwrap();
        unsafe { cuda_free(&ptr) }.unwrap();
    }

    #[test]
    fn test_cuda_mem_cpy() {
        let len = 1 << 10;

        let mut rng = thread_rng();

        let mut ptr: DevicePointer<u32> = unsafe { cuda_malloc(len) }.unwrap();

        let host_values = (0..len).map(|_| rng.gen::<u32>()).collect::<Vec<_>>();
        // Copy values from host to device.
        unsafe { copy_host_to_device(&mut ptr, &host_values, len) }.unwrap();

        // Allocate a new vector for the values
        let mut device_values = vec![0u32; len];
        // Copy the values from device.
        unsafe { copy_device_to_host(&mut device_values, &ptr, len) }.unwrap();
        // Compare to original values.
        unsafe { cuda_free(&ptr) }.unwrap();

        for (val, exp) in host_values.iter().zip(device_values.iter()) {
            assert_eq!(val, exp);
        }
    }
}
