use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

use crate::{
    device::{error::CudaError, DeviceBuffer},
    matrix::ColMajorMatrixDevice,
};

use super::*;

impl DeviceBuffer<BabyBear> {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn sum(&self, result: *mut BabyBear) -> Result<(), CudaError> {
        unsafe {
            ffi::sum_baby_bear(self.as_ptr(), result, 1, self.len(), self.stream().handle())
                .to_result()
        }
    }
}

impl ColMajorMatrixDevice<BabyBear> {
    pub fn column_sum(&self, results: &mut DeviceBuffer<BabyBear>) -> Result<(), CudaError> {
        unsafe {
            ffi::sum_baby_bear(
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

impl DeviceBuffer<BinomialExtensionField<BabyBear, 4>> {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn sum(&self, result: *mut BinomialExtensionField<BabyBear, 4>) -> Result<(), CudaError> {
        unsafe {
            ffi::sum_baby_bear_extension(
                self.as_ptr(),
                result,
                1,
                self.len(),
                self.stream().handle(),
            )
            .to_result()
        }
    }
}

impl ColMajorMatrixDevice<BinomialExtensionField<BabyBear, 4>> {
    pub fn column_sum(
        &self,
        results: &mut DeviceBuffer<BinomialExtensionField<BabyBear, 4>>,
    ) -> Result<(), CudaError> {
        unsafe {
            ffi::sum_baby_bear_extension(
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

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, FieldAlgebra};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use rand::{thread_rng, Rng};

    use crate::{
        device::{
            memory::{ToDevice, ToHost},
            DeviceBuffer,
        },
        matrix::ColMajorMatrixDevice,
    };
    use rayon::prelude::*;

    #[test]
    fn test_sum_baby_bear() {
        let mut rng = thread_rng();

        for input_log_size in 12..31 {
            println!("Input log size: {}", input_log_size);
            let input_size = 1 << input_log_size;
            let input_host = (0..input_size).map(|_| rng.gen::<BabyBear>()).collect::<Vec<_>>();

            let input_device = input_host.to_device().unwrap();

            assert_eq!(input_device.len(), input_size);

            let mut result = [BabyBear::ZERO].to_device().unwrap();
            assert_eq!(result.len(), 1);

            input_device.stream().synchronize().unwrap();
            let time = Instant::now();
            input_device.sum(result.as_mut_ptr()).unwrap();
            input_device.stream().synchronize().unwrap();
            let elapsed = time.elapsed();
            println!("Device time: {:?}", elapsed);

            let sum_device = result.to_host()[0];

            let time = Instant::now();
            let sum_host = input_host.iter().copied().sum::<BabyBear>();
            let elapsed = time.elapsed();
            println!("host time: {:?}", elapsed);

            assert_eq!(sum_device, sum_host);

            println!("------------------------");
        }
    }

    #[test]
    fn test_column_sum_baby_bear() {
        let mut rng = thread_rng();

        for input_log_height in 12..24 {
            for width in [50, 100, 150, 200] {
                println!("(Input log height, input width) : ({}, {})", input_log_height, width);
                let input_height = 1 << input_log_height;
                let input_host =
                    (0..input_height * width).map(|_| rng.gen::<BabyBear>()).collect::<Vec<_>>();

                let input_host = RowMajorMatrix::new(input_host, width);

                let input_device = input_host.to_device().unwrap().to_column_major();

                assert_eq!(input_device.height(), input_height);

                let mut results = DeviceBuffer::<BabyBear>::with_capacity(width).unwrap();
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
                let mut column_sums = vec![BabyBear::ZERO; width];
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
    fn test_column_sum_extension_baby_bear() {
        let mut rng = thread_rng();

        type EF = BinomialExtensionField<BabyBear, 4>;
        for input_log_height in 16..22 {
            for width in [10, 50, 100, 200] {
                println!("(Input log height, input width) : ({}, {})", input_log_height, width);
                let input_height = 1 << input_log_height;
                let input_host =
                    (0..input_height * width).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();

                let input_host = RowMajorMatrix::new(input_host, width);

                let input_device_values =
                    input_host.clone().transpose().values.to_device().unwrap();
                let input_device = ColMajorMatrixDevice::new(input_device_values, input_height);

                assert_eq!(input_device.height(), input_height);
                assert_eq!(input_device.width(), width);

                let mut results = DeviceBuffer::<EF>::with_capacity(width).unwrap();
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
                let mut column_sums = vec![EF::ZERO; width];
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
    fn test_sum_extension_baby_bear() {
        let mut rng = thread_rng();

        for input_log_size in 12..29 {
            println!("Input log size: {}", input_log_size);
            let input_size = 1 << input_log_size;
            let input_host = (0..input_size)
                .map(|_| rng.gen::<BinomialExtensionField<BabyBear, 4>>())
                .collect::<Vec<_>>();

            let input_device = input_host.to_device().unwrap();

            assert_eq!(input_device.len(), input_size);

            let mut result = [BinomialExtensionField::<BabyBear, 4>::ZERO].to_device().unwrap();
            assert_eq!(result.len(), 1);

            input_device.stream().synchronize().unwrap();
            let time = Instant::now();
            input_device.sum(result.as_mut_ptr()).unwrap();
            input_device.stream().synchronize().unwrap();
            let elapsed = time.elapsed();
            println!("Device time: {:?}", elapsed);

            let sum_device = result.to_host()[0];

            let time = Instant::now();
            let sum_host =
                input_host.par_iter().copied().sum::<BinomialExtensionField<BabyBear, 4>>();
            let elapsed = time.elapsed();
            println!("host time: {:?}", elapsed);

            assert_eq!(sum_device, sum_host);

            println!("------------------------");
        }
    }
}
