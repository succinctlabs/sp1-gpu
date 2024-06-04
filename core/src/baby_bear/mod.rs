// use p3_baby_bear::BabyBear;
// use p3_field::extension::BinomialExtensionField;

// use crate::device::{error::CudaError, slice::DeviceSlice};

// type F = BabyBear;
// const D: usize = 4;
// type EF = BinomialExtensionField<F, D>;

// mod ffi {
//     use super::*;
//     use crate::device::error::CudaRustError;

//     extern "C" {
//         pub fn scan_baby_bear(a: *const F, b: *const F, n: usize) -> CudaRustError;
//         pub fn scan_baby_bear_challenge(a: *const EF, b: *const EF, n: usize) -> CudaRustError;
//     }
// }

// impl DeviceSlice<F> {
//     pub fn scan(&self, result: &mut Self) -> Result<(), CudaError> {
//         assert_eq!(result.len(), self.len());
//         unsafe { ffi::scan_baby_bear(self.as_ptr(), result.as_mut_ptr(), self.len()).to_result() }
//     }
// }

// impl DeviceSlice<EF> {
//     pub fn scan(&self, result: &mut Self) -> Result<(), CudaError> {
//         assert_eq!(result.len(), self.len());
//         unsafe {
//             ffi::scan_baby_bear_challenge(self.as_ptr(), result.as_mut_ptr(), self.len())
//                 .to_result()
//         }
//     }
// }
