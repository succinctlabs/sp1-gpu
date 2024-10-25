use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, TwoAdicField};

use crate::{
    device::{error::CudaError, DeviceBuffer},
    matrix::ColMajorMatrixDevice,
};

pub(super) mod ffi;

pub fn subgroup_normalizer<F: TwoAdicField>(log_order: usize) -> F {
    let domain_generator = F::two_adic_generator(log_order);
    let mut domain_point = F::one();
    let mut domain_normalizer = F::one();
    for _ in 1..log_order {
        domain_point *= domain_generator;
        domain_normalizer *= F::one() - domain_point;
    }
    domain_normalizer.inverse()
}

impl ColMajorMatrixDevice<BabyBear> {
    pub fn eval(
        &self,
        results: &mut DeviceBuffer<BinomialExtensionField<BabyBear, 4>>,
        normalizer: BabyBear,
        evaluation_point: BinomialExtensionField<BabyBear, 4>,
        vanishing_poly_eval: BinomialExtensionField<BabyBear, 4>,
    ) -> Result<(), CudaError> {
        let log_height = self.height.ilog2() as usize;
        unsafe {
            ffi::univariate_eval_babybear(
                results.as_mut_ptr(),
                self.values.as_ptr(),
                BabyBear::two_adic_generator(log_height),
                normalizer,
                evaluation_point,
                vanishing_poly_eval,
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
    use std::time::Instant;

    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, AbstractField};
    use p3_matrix::dense::RowMajorMatrix;
    use rand::{thread_rng, Rng};

    use crate::{
        device::{memory::ToDevice, DeviceBuffer},
        univariate::subgroup_normalizer,
    };

    #[test]
    fn test_univariate_evaluation_babybear() {
        let mut rng = thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;

        for input_log_height in 16..26 {
            let input_height = 1 << input_log_height;
            let domain_normalizer = subgroup_normalizer(input_log_height);

            for width in [1, 50, 100, 200] {
                if input_log_height > 22 && width > 50 {
                    continue;
                }
                println!("(Input log height, input width) : ({}, {})", input_log_height, width);
                let input_host =
                    (0..input_height * width).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

                let input_host = RowMajorMatrix::new(input_host, width);

                let input_device = input_host.to_device().unwrap().to_column_major();

                assert_eq!(input_device.height(), input_height);

                let mut results = DeviceBuffer::<EF>::with_capacity(width).unwrap();
                unsafe {
                    results.set_max_len();
                }
                assert_eq!(results.len(), width);

                let point: EF = rng.gen();
                let time = Instant::now();
                let vanishing_poly = point.exp_power_of_2(input_log_height) - EF::one();
                let elapsed = time.elapsed();
                println!("Time to compute vanishing poly: {:?}", elapsed);

                input_device.stream().synchronize().unwrap();
                let time = Instant::now();
                input_device.eval(&mut results, domain_normalizer, point, vanishing_poly).unwrap();
                input_device.stream().synchronize().unwrap();
                let elapsed = time.elapsed();
                println!("Device time: {:?}", elapsed);
            }
        }
    }
}
