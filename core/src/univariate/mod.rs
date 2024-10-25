use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, TwoAdicField};

use crate::{
    device::{error::CudaError, DeviceBuffer},
    matrix::ColMajorMatrixDevice,
};

pub(super) mod ffi;

impl ColMajorMatrixDevice<BabyBear> {
    pub fn eval(
        &self,
        results: &mut DeviceBuffer<BinomialExtensionField<BabyBear, 4>>,
        normalizer: BabyBear,
        point: BinomialExtensionField<BabyBear, 4>,
    ) -> Result<(), CudaError> {
        let log_height = self.height.ilog2() as usize;
        unsafe {
            ffi::univariate_eval_babybear(
                results.as_mut_ptr(),
                self.values.as_ptr(),
                BabyBear::two_adic_generator(log_height),
                normalizer,
                point,
                self.width(),
                log_height,
                self.stream().handle(),
            )
            .to_result()
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_univariate_evaluation() {}
}
