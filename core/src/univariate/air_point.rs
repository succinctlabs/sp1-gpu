use std::ops::{Add, AddAssign};

use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

use crate::{
    device::{error::CudaError, DeviceBuffer},
    matrix::ColMajorMatrixDevice,
};

use super::ffi;

/// A struct that houses a pair of points.
///
/// This struct is useful for getting combined methods on a view of an AIR, consisting of values
/// in the `local` and `next` rows.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct AirPoint<T> {
    local: T,
    next: T,
}

impl<T> AirPoint<T> {
    #[inline]
    pub const fn new(local: T, next: T) -> Self {
        AirPoint { local, next }
    }
}

impl ColMajorMatrixDevice<AirPoint<BinomialExtensionField<BabyBear, 4>>> {
    pub fn column_sum(
        &self,
        results: &mut DeviceBuffer<AirPoint<BinomialExtensionField<BabyBear, 4>>>,
    ) -> Result<(), CudaError> {
        unsafe {
            ffi::sum_air_point_baby_bear_extension(
                self.values.as_ptr(),
                results.as_mut_ptr(),
                self.width(),
                self.height,
                self.stream().handle(),
            )
            .to_result()
        }
    }
}

impl<T, Rhs> AddAssign<AirPoint<Rhs>> for AirPoint<T>
where
    T: AddAssign<Rhs>,
{
    fn add_assign(&mut self, rhs: AirPoint<Rhs>) {
        self.local += rhs.local;
        self.next += rhs.next;
    }
}

impl<T, Rhs> Add<AirPoint<Rhs>> for AirPoint<T>
where
    T: Add<Rhs>,
{
    type Output = AirPoint<<T as Add<Rhs>>::Output>;

    fn add(self, rhs: AirPoint<Rhs>) -> Self::Output {
        AirPoint::new(self.local + rhs.local, self.next + rhs.next)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_challenger::CanObserve;
    use p3_commit::Pcs;
    use p3_field::{extension::BinomialExtensionField, AbstractField, TwoAdicField};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::{thread_rng, Rng};
    use sp1_stark::{baby_bear_poseidon2::BabyBearPoseidon2, StarkGenericConfig};

    use crate::{
        device::{
            memory::{ToDevice, ToHost},
            DeviceBuffer,
        },
        matrix::ColMajorMatrixDevice,
        univariate::{subgroup_normalizer, AirPoint},
    };

    #[test]
    fn test_column_sum_extension_air_point_baby_bear() {
        let mut rng = thread_rng();

        type EF = BinomialExtensionField<BabyBear, 4>;
        for input_log_height in 16..22 {
            for width in [10, 50, 100, 200] {
                println!("(Input log height, input width) : ({}, {})", input_log_height, width);
                let input_height = 1 << input_log_height;
                let input_host = (0..input_height * width)
                    .map(|_| AirPoint::new(rng.gen::<EF>(), rng.gen::<EF>()))
                    .collect::<Vec<_>>();

                let input_host = RowMajorMatrix::new(input_host, width);

                let input_device_values =
                    input_host.clone().transpose().values.to_device().unwrap();
                let input_device = ColMajorMatrixDevice::new(input_device_values, input_height);

                assert_eq!(input_device.height(), input_height);
                assert_eq!(input_device.width(), width);

                let mut results = DeviceBuffer::<AirPoint<EF>>::with_capacity(width).unwrap();
                unsafe {
                    results.set_max_len();
                }
                assert_eq!(results.len(), width);

                input_device.stream().synchronize().unwrap();
                let time = Instant::now();
                input_device.column_sum(&mut results).unwrap();
                input_device.stream().synchronize().unwrap();
                let elapsed = time.elapsed();
                println!("Device time: {:?}", elapsed);

                let sums_device = results.to_host();

                let time = Instant::now();
                let mut column_sums = vec![AirPoint::new(EF::zero(), EF::zero()); width];
                for row in input_host.rows() {
                    for (element, accum) in row.zip_eq(column_sums.iter_mut()) {
                        *accum += element;
                    }
                }
                let elapsed = time.elapsed();
                println!("host time: {:?}", elapsed);

                for (i, (elem, expected)) in sums_device.into_iter().zip(column_sums).enumerate() {
                    assert_eq!(elem, expected, "failed at column {}", i);
                }

                println!("------------------------");
            }
        }
    }

    #[test]
    fn test_univariate_evaluation_air_point_babybear() {
        let mut rng = thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;
        type SC = BabyBearPoseidon2;

        for input_log_height in 16..26 {
            let input_height = 1 << input_log_height;
            let domain_normalizer = subgroup_normalizer(input_log_height);
            println!("domain normalizer {}", domain_normalizer);

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

                let mut results = DeviceBuffer::<AirPoint<EF>>::with_capacity(width).unwrap();
                unsafe {
                    results.set_max_len();
                }
                assert_eq!(results.len(), width);

                let point: EF = rng.gen();
                input_device.stream().synchronize().unwrap();
                let time = Instant::now();
                let vanishing_poly = point.exp_power_of_2(input_log_height) - EF::one();
                input_device
                    .eval_air_point(&mut results, domain_normalizer, point, vanishing_poly)
                    .unwrap();
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

                    let g = BabyBear::two_adic_generator(input_log_height);
                    let (openings, _) = <<SC as StarkGenericConfig>::Pcs as Pcs<
                        <SC as StarkGenericConfig>::Challenge,
                        <SC as StarkGenericConfig>::Challenger,
                    >>::open(
                        config.pcs(),
                        vec![(&prover_data, vec![vec![point, point * g]; width])],
                        &mut challenger,
                    );
                    let [openings_for_round] = openings.try_into().unwrap();
                    let [openings_for_mat] = openings_for_round.try_into().unwrap();
                    let [openings_for_point, openings_for_next_point] =
                        openings_for_mat.try_into().unwrap();

                    let device_openings = results.to_host();
                    for (idx, ((value, expected_local), expected_next)) in device_openings
                        .into_iter()
                        .zip_eq(openings_for_point)
                        .zip_eq(openings_for_next_point)
                        .enumerate()
                    {
                        assert_eq!(value.local, expected_local, "failed at column {}", idx);
                        assert_eq!(value.next, expected_next, "failed at column {}", idx);
                    }
                }
            }
        }
    }
}
