use std::{
    alloc::Layout,
    ffi::c_void,
    mem,
    ptr::{self, NonNull},
};

use moongate_bloc::alloc::{AllocError, Allocator, DeviceMemory};

use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaError, ffi},
};

use super::buffer::DeviceBuffer;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GlobalDeviceAllocator;

impl DeviceMemory for GlobalDeviceAllocator {
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
    ) -> Result<(), AllocError> {
        copy_device_to_device(dst, src, size).map_err(|_| AllocError)
    }

    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), AllocError> {
        cuda_mem_set_(dst, value, size).map_err(|_| AllocError)
    }
}

unsafe impl Allocator for GlobalDeviceAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let len = layout.size();
            let ptr = cuda_malloc::<u8>(len).map_err(|_| AllocError)?;
            Ok(NonNull::slice_from_raw_parts(NonNull::new_unchecked(ptr), len))
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        cuda_free(ptr.as_ptr()).unwrap()
    }
}

pub trait ToDevice {
    type DeviceType;

    fn to_device(&self) -> Result<Self::DeviceType, CudaError> {
        self.to_device_async(&CudaStream::default())
    }

    fn to_device_async(&self, stream: &CudaStream) -> Result<Self::DeviceType, CudaError>;
}

pub trait ToHost {
    type HostType;

    fn to_host(&self) -> Self::HostType;
}

impl<T: Copy> ToDevice for [T] {
    type DeviceType = DeviceBuffer<T>;

    fn to_device_async(&self, stream: &CudaStream) -> Result<Self::DeviceType, CudaError> {
        let mut buffer = DeviceBuffer::with_capacity_in(self.len(), stream)?;
        buffer.extend_from_host_slice(self);
        Ok(buffer)
    }
}

/// A Rust interface for cudaMalloc.
///
/// # Safety
/// The pointer will not be dropped unless a `cudaFree` call is initiated.
pub unsafe fn cuda_malloc<T>(len: usize) -> Result<*mut T, CudaError> {
    let mut ptr: *mut c_void = ptr::null_mut();
    Result::<(), CudaError>::from(ffi::cuda_malloc(
        &mut ptr as *mut *mut c_void,
        len * mem::size_of::<T>(),
    ))?;

    Ok(ptr as *mut T)
}

/// A Rust interface for cudaMallocHost.
///
/// Allocates pinned memory on the host.
///
/// # Safety
/// The memory will not be freed until `cudaFreeHost` is called.
pub unsafe fn cuda_malloc_host<T>(len: usize) -> Result<*mut T, CudaError> {
    let mut ptr: *mut c_void = ptr::null_mut();
    Result::<(), CudaError>::from(ffi::cuda_malloc_host(
        &mut ptr as *mut *mut c_void,
        len * mem::size_of::<T>(),
    ))?;

    Ok(ptr as *mut T)
}

/// # Safety
/// The memory will not be freed until `cudaFreeHost` is called.
pub unsafe fn cuda_mem_set_<T>(ptr: *const T, value: u8, len: usize) -> Result<(), CudaError> {
    ffi::cuda_mem_set(ptr as *mut c_void, value, len * mem::size_of::<T>()).to_result()
}

/// A Rust interface for cudaHostRegister.
///
/// Registers host memory as pinned for cuda usage.
///
/// # Safety
/// The memory will not be unregistered until `cudaHostUnregister` is called.
pub unsafe fn cuda_host_register<T>(ptr: *const T, len: usize) -> Result<(), CudaError> {
    Result::<(), CudaError>::from(ffi::cuda_host_register(
        ptr as *const c_void,
        len * mem::size_of::<T>(),
    ))
}

/// A Rust interface for cudaFreeHost.
///
/// # Safety
/// The caller must guarantee that after this call no data will point to the value of the pointer.
pub unsafe fn cuda_free_host<T>(ptr: *const T) -> Result<(), CudaError> {
    Result::<(), CudaError>::from(ffi::cuda_free_host(ptr as *const c_void))
}

/// A Rust interface for cudaHostUnregister.
///
/// # Safety
/// The caller must guarantee that after this call no data will point to the value of the pointer.
pub unsafe fn cuda_host_unregister<T>(ptr: *const T) -> Result<(), CudaError> {
    Result::<(), CudaError>::from(ffi::cuda_host_unregister(ptr as *const c_void))
}

/// A Rust interface for cudaFree.
///
/// # Safety
/// The caller must guarantee that after this call no data will point to the value of the pointer.
pub unsafe fn cuda_free<T>(ptr: *mut T) -> Result<(), CudaError> {
    ffi::cuda_free(ptr as *mut c_void).into()
}

/// A Rust interface for cudaMemcpy from host to device.
///
/// # Safety
/// The caller must guarantee that the memory copied from/into size is valid.
pub unsafe fn copy_host_to_device<T: Copy>(
    dst: *mut T,
    src: *const T,
    len: usize,
) -> Result<(), CudaError> {
    ffi::cuda_mem_copy_host_to_device(
        dst as *mut c_void,
        src as *const c_void,
        len * mem::size_of::<T>(),
    )
    .into()
}

/// A Rust interface for cudaMemcpy from device to host.
///
/// # Safety
/// The caller must guarantee that the memory copied from/into size is valid.
pub unsafe fn copy_device_to_host<T: Copy>(
    dst: *mut T,
    src: *const T,
    len: usize,
) -> Result<(), CudaError> {
    ffi::cuda_mem_copy_device_to_host(
        dst as *mut c_void,
        src as *const c_void,
        len * mem::size_of::<T>(),
    )
    .into()
}

/// A Rust interface for cudaMemcpy.
///
/// # Safety
/// The caller must guarantee that the memory copied from/into size is valid.
pub unsafe fn copy_device_to_device<T: Copy>(
    dst: *mut T,
    src: *const T,
    len: usize,
) -> Result<(), CudaError> {
    ffi::cuda_mem_copy_device_to_device(
        dst as *mut c_void,
        src as *const c_void,
        len * mem::size_of::<T>(),
    )
    .into()
}

/// Returns a pair `(free, total)` of the amount of free and total memory on the device.
pub fn cuda_mem_get_info() -> Result<(usize, usize), CudaError> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe { ffi::cuda_mem_get_info(&mut free, &mut total) }.to_result()?;
    Ok((free, total))
}

#[cfg(test)]
mod tests {

    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn test_cuda_malloc() {
        let len = 1 << 4;
        let ptr: *mut u32 = unsafe { cuda_malloc(len) }.unwrap();
        unsafe { cuda_free(ptr) }.unwrap();
    }

    #[test]
    fn test_cuda_mem_cpy() {
        let len = 1 << 10;

        let mut rng = thread_rng();

        let ptr: *mut u32 = unsafe { cuda_malloc(len) }.unwrap();

        let host_values = (0..len).map(|_| rng.gen::<u32>()).collect::<Vec<_>>();
        // Copy values from host to device.
        unsafe { copy_host_to_device(ptr, host_values.as_ptr(), len) }.unwrap();

        // Allocate a new vector for the values
        let mut device_values = vec![0u32; len];
        // Copy the values from device.
        unsafe { copy_device_to_host(device_values.as_mut_ptr(), ptr, len) }.unwrap();
        // Compare to original values.
        unsafe { cuda_free(ptr) }.unwrap();

        for (val, exp) in host_values.iter().zip(device_values.iter()) {
            assert_eq!(val, exp);
        }
    }
}
