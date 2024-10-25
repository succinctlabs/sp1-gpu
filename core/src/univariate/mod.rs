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
    for _ in 1..(1 << log_order) {
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

    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_challenger::CanObserve;
    use p3_commit::Pcs;
    use p3_field::{extension::BinomialExtensionField, AbstractField};
    use p3_matrix::dense::RowMajorMatrix;
    use rand::{thread_rng, Rng};
    use sp1_stark::{baby_bear_poseidon2::BabyBearPoseidon2, StarkGenericConfig};

    use crate::{
        device::{
            memory::{ToDevice, ToHost},
            DeviceBuffer,
        },
        univariate::subgroup_normalizer,
    };

    #[test]
    fn test_univariate_evaluation_babybear() {
        let mut rng = thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;
        type SC = BabyBearPoseidon2;

        for input_log_height in 16..26 {
            let input_height = 1 << input_log_height;
            let domain_normalizer = subgroup_normalizer(input_log_height);
            println!("domain normalizer {}", domain_normalizer);

            for width in [1, 50, 100, 200, 400] {
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
                input_device.stream().synchronize().unwrap();
                let time = Instant::now();
                let vanishing_poly = point.exp_power_of_2(input_log_height) - EF::one();
                input_device.eval(&mut results, domain_normalizer, point, vanishing_poly).unwrap();
                input_device.stream().synchronize().unwrap();
                let elapsed = time.elapsed();
                println!("Device time: {:?}", elapsed);

                // If the input height is small enough, compare the evaluations to cpu values.
                if input_log_height < 18 {
                    let config = SC::default();
                    let domain = <<SC as StarkGenericConfig>::Pcs as Pcs<
                        <SC as StarkGenericConfig>::Challenge,
                        <SC as StarkGenericConfig>::Challenger,
                    >>::natural_domain_for_degree(
                        config.pcs(), input_height
                    );
                    let mut challenger = config.challenger();
                    let (commit, prover_data) =
                        <<SC as StarkGenericConfig>::Pcs as Pcs<
                            <SC as StarkGenericConfig>::Challenge,
                            <SC as StarkGenericConfig>::Challenger,
                        >>::commit(config.pcs(), vec![(domain, input_host)]);
                    challenger.observe(commit);

                    let (openings, _) = <<SC as StarkGenericConfig>::Pcs as Pcs<
                        <SC as StarkGenericConfig>::Challenge,
                        <SC as StarkGenericConfig>::Challenger,
                    >>::open(
                        config.pcs(),
                        vec![(&prover_data, vec![vec![point]; width])],
                        &mut challenger,
                    );
                    let [openings_for_round] = openings.try_into().unwrap();
                    let [openings_for_mat] = openings_for_round.try_into().unwrap();
                    let [openings_for_point] = openings_for_mat.try_into().unwrap();

                    let device_openings = results.to_host();
                    for (idx, (value, expected)) in
                        device_openings.into_iter().zip_eq(openings_for_point).enumerate()
                    {
                        assert_eq!(value, expected, "failed at column {}", idx);
                    }
                }
            }
        }
    }
}
