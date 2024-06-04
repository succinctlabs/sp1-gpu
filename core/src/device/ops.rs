use super::{
    buffer::DeviceBuffer,
    error::{CudaError, CudaRustError},
    slice::DeviceSlice,
};

pub trait CudaScan: Copy {
    /// # Safety
    ///
    /// TODO
    unsafe fn cuda_scan(a: *mut Self, b: *const Self, n: usize) -> CudaRustError;
}

impl<T> DeviceSlice<T> {
    pub fn scan_into(&self, result: &mut Self) -> Result<(), CudaError>
    where
        T: CudaScan,
    {
        assert_eq!(result.len(), self.len());
        unsafe { T::cuda_scan(result.as_mut_ptr(), self.as_ptr(), self.len()).to_result() }
    }

    pub fn scan_inplace(&mut self) -> Result<(), CudaError>
    where
        T: CudaScan,
    {
        unsafe { T::cuda_scan(self.as_mut_ptr(), self.as_ptr(), self.len()).to_result() }
    }
}

impl<T: Copy> DeviceBuffer<T> {
    pub fn scan(&self) -> Result<Self, CudaError>
    where
        T: CudaScan,
    {
        let mut result = Self::with_capacity(self.len());
        unsafe { result.set_max_len() };
        self.scan_into(&mut result)?;
        Ok(result)
    }

    pub fn scan_inplace(&mut self) -> Result<(), CudaError>
    where
        T: CudaScan,
    {
        self.as_slice_mut().scan_inplace()
    }
}
