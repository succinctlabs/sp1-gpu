use super::{
    error::{CudaError, CudaRustError},
    Buffer, RawDevicePointer,
};

pub trait CudaScan: Copy {
    /// # Safety
    ///
    /// TODO
    unsafe fn cuda_scan(a: *mut Self, b: *const Self, n: usize) -> CudaRustError;
}

impl<P: RawDevicePointer> Buffer<P>
where
    P::Data: CudaScan,
{
    pub fn scan(&self) -> Result<Self, CudaError> {
        let mut result = Self::with_capacity_in(self.len(), self.allocator())?;
        unsafe { result.set_max_len() };
        unsafe { P::Data::cuda_scan(result.as_mut_ptr(), self.as_ptr(), self.len()).to_result() }?;
        // self.scan_into(&mut result)?;
        Ok(result)
    }

    pub fn scan_inplace(&mut self) -> Result<(), CudaError> {
        unsafe { P::Data::cuda_scan(self.as_mut_ptr(), self.as_ptr(), self.len()).to_result() }
    }
}
