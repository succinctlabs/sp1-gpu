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
    use p3_field::{extension::BinomialExtensionField, AbstractField};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::{thread_rng, Rng};

    use crate::{
        device::{
            memory::{ToDevice, ToHost},
            DeviceBuffer,
        },
        matrix::ColMajorMatrixDevice,
        univariate::AirPoint,
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
}
