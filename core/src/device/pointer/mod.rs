mod pinned;
mod stream;
mod sync;

pub use pinned::*;
pub use stream::*;
pub use sync::*;

use super::error::CudaError;

pub trait RawPointer {
    type Data;

    fn as_ptr(&self) -> *const Self::Data;
    fn as_mut_ptr(&mut self) -> *mut Self::Data;

    fn free(&mut self);
}

pub trait CopyRawFrom<P> {
    /// # Safety
    ///
    /// The pointer must be valid, it must have allocated memory in the size of `len`
    unsafe fn copy_from(&mut self, src: &P, len: usize) -> Result<(), CudaError>;
}

pub trait CopyRawTo<P> {
    /// # Safety
    ///
    /// The pointer must be valid, it must have allocated memory in the size of `len`
    unsafe fn copy_to(&self, dst: &mut P, len: usize) -> Result<(), CudaError>;
}

impl<T, S> CopyRawTo<S> for T
where
    S: CopyRawFrom<T>,
{
    unsafe fn copy_to(&self, dst: &mut S, len: usize) -> Result<(), CudaError> {
        dst.copy_from(self, len)
    }
}
