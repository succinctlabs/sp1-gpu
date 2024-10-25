use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

use crate::{cuda_runtime::stream::CudaStreamHandle, device::error::CudaRustError};

use super::AirPoint;

extern "C" {
    #[link_name = "evalUnivariateBabyBear"]
    pub(crate) fn univariate_eval_babybear(
        results: *mut BinomialExtensionField<BabyBear, 4>,
        polynomial_batch: *const BabyBear,
        domain_generator: BabyBear,
        domain_normalizer: BabyBear,
        evaluation_point: BinomialExtensionField<BabyBear, 4>,
        vanishing_poly_eval: BinomialExtensionField<BabyBear, 4>,
        width: usize,
        log_height: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    #[link_name = "vectorsumAirBabyBearExtension"]
    pub fn sum_air_point_baby_bear_extension(
        input: *const AirPoint<BinomialExtensionField<BabyBear, 4>>,
        result: *mut AirPoint<BinomialExtensionField<BabyBear, 4>>,
        width: usize,
        height: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}
