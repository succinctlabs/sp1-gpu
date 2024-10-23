use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

use crate::{cuda_runtime::stream::CudaStreamHandle, device::error::CudaRustError};

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
        input: *mut BabyBear,
        result: *mut BabyBear,
        len: usize,
        stream: CudaStreamHandle,
    );

    #[link_name = "vectorSumBabyBearExtension"]
    pub fn sum_baby_bear_extension(
        input: *mut BinomialExtensionField<BabyBear, 4>,
        result: *mut BinomialExtensionField<BabyBear, 4>,
        len: usize,
        stream: CudaStreamHandle,
    );

    #[link_name = "partialSumBabyBear"]
    pub fn partial_sum_baby_bear(
        input: *mut BabyBear,
        output: *mut BabyBear,
        len: usize,
        stream: CudaStreamHandle,
    );
}
