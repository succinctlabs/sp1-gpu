mod pinned;
mod stream;
mod sync;

use std::ffi::c_void;

pub use pinned::*;
pub use stream::*;
pub use sync::*;

use crate::cuda_runtime::stream::CudaStream;

use super::{error::CudaError, DeviceAllocator};

pub trait Offset {
    /// # Safety
    ///
    /// The caller must guarantee that the offset is valid.
    unsafe fn add(&self, offset: usize) -> Self;
}

pub trait RawPointer: Sized + Offset {
    type Data;

    fn as_ptr(&self) -> *const Self::Data;
    fn as_mut_ptr(&mut self) -> *mut Self::Data;

    fn free(&mut self);
}

pub trait DefaultAllocatorPointer: RawPointer {
    type Allocator: DeviceAllocator<Self>;

    fn allocator(&self) -> &Self::Allocator;
}

pub trait RawDevicePointer:
    DefaultAllocatorPointer
    + CopyRawFrom<Self>
    + CopyRawTo<*mut Self::Data>
    + CopyRawFrom<*const Self::Data>
{
    fn sync(&self) -> Result<(), CudaError>;

    fn stream_raw(&self) -> *mut c_void;
}

pub trait CopyRawFrom<P> {
    /// # Safety
    ///
    /// The pointer must be valid, it must have allocated memory in the size of `len`
    unsafe fn copy_raw_from(&mut self, src: &P, len: usize) -> Result<(), CudaError>;
}

pub trait CopyRawTo<P> {
    /// # Safety
    ///
    /// The pointer must be valid, it must have allocated memory in the size of `len`
    unsafe fn copy_raw_to(&self, dst: &mut P, len: usize) -> Result<(), CudaError>;
}

impl<T, S> CopyRawTo<S> for T
where
    S: CopyRawFrom<T>,
{
    unsafe fn copy_raw_to(&self, dst: &mut S, len: usize) -> Result<(), CudaError> {
        dst.copy_raw_from(self, len)
    }
}
