use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

use crate::{cuda_runtime::stream::CudaStreamHandle, device::error::CudaRustError};

/// cbindgen:ignore
extern "C" {
    pub fn scan_baby_bear(
        a: *const BabyBear,
        b: *const BabyBear,
        n: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    pub fn scan_baby_bear_challenge(
        a: *const BinomialExtensionField<BabyBear, 4>,
        b: *const BinomialExtensionField<BabyBear, 4>,
        n: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    #[link_name = "vectorSumBabyBear"]
    pub fn sum_baby_bear(
        input: *const BabyBear,
        result: *mut BabyBear,
        width: usize,
        height: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    #[link_name = "vectorSumBabyBearExtension"]
    pub fn sum_baby_bear_extension(
        input: *const BinomialExtensionField<BabyBear, 4>,
        result: *mut BinomialExtensionField<BabyBear, 4>,
        width: usize,
        height: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

}
