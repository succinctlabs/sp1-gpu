use super::error::CudaRustError;

pub trait IterScan: Copy {
    /// # Safety
    ///
    /// TODO
    unsafe fn cuda_scan(a: *mut Self, b: *const Self, n: usize) -> CudaRustError;
}
