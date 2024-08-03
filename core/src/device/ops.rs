use super::{
    error::{CudaError, CudaRustError},
    Buffer, DefaultAllocatorPointer,
};

pub trait Scan: Copy {
    /// # Safety
    ///
    /// TODO
    unsafe fn scan_raw(a: *mut Self, b: *const Self, n: usize) -> CudaRustError;
}

impl<P: DefaultAllocatorPointer> Buffer<P>
where
    P::Data: Scan,
{
    pub fn scan(&self) -> Result<Self, CudaError> {
        let mut result = Self::with_capacity_in(self.len(), self.allocator())?;
        unsafe { result.set_max_len() };
        unsafe { P::Data::scan_raw(result.as_mut_ptr(), self.as_ptr(), self.len()).to_result() }?;
        // self.scan_into(&mut result)?;
        Ok(result)
    }

    pub fn scan_inplace(&mut self) -> Result<(), CudaError> {
        unsafe { P::Data::scan_raw(self.as_mut_ptr(), self.as_ptr(), self.len()).to_result() }
    }
}
