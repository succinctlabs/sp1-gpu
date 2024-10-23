use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

use crate::device::DeviceBuffer;

use super::*;

impl DeviceBuffer<BabyBear> {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn sum(&self, result: *mut BabyBear) {
        unsafe {
            ffi::sum_baby_bear(
                self.as_ptr() as *mut BabyBear,
                result,
                self.len(),
                self.stream().handle(),
            )
        }
    }
}

impl DeviceBuffer<BinomialExtensionField<BabyBear, 4>> {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn sum(&self, result: *mut BinomialExtensionField<BabyBear, 4>) {
        unsafe {
            ffi::sum_baby_bear_extension(
                self.as_ptr() as *mut BinomialExtensionField<BabyBear, 4>,
                result,
                self.len(),
                self.stream().handle(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, AbstractField};
    use rand::{thread_rng, Rng};

    use crate::device::{
        memory::{ToDevice, ToHost},
        DeviceBuffer,
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

            let mut result = [BabyBear::zero()].to_device().unwrap();
            assert_eq!(result.len(), 1);

            input_device.stream().synchronize().unwrap();
            let time = Instant::now();
            input_device.sum(result.as_mut_ptr());
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

            let mut result = [BinomialExtensionField::<BabyBear, 4>::zero()].to_device().unwrap();
            assert_eq!(result.len(), 1);

            input_device.stream().synchronize().unwrap();
            let time = Instant::now();
            input_device.sum(result.as_mut_ptr());
            input_device.stream().synchronize().unwrap();
            let elapsed = time.elapsed();
            println!("Device time: {:?}", elapsed);

            let sum_device = result.to_host()[0];

            let time = Instant::now();
            let sum_host =
                input_host.par_iter().copied().sum::<BinomialExtensionField<BabyBear, 4>>();
            let elapsed = time.elapsed();
            println!("host time: {:?}", elapsed);

            // assert_eq!(sum_device, sum_host);

            println!("------------------------");
        }
    }

    // #[test]
    // fn test_partial_sum_baby_bear() {
    //     let mut rng = thread_rng();

    //     for input_log_size in 10..31 {
    //         println!("Input log size: {}", input_log_size);
    //         let input_size = 1 << input_log_size;
    //         let input_host = (0..input_size).map(|_| rng.gen::<BabyBear>()).collect::<Vec<_>>();

    //         let input_device = input_host.to_device().unwrap();

    //         assert_eq!(input_device.len(), input_size);

    //         let num_blocks = input_size.div_ceil(512);
    //         let mut partial_sums = DeviceBuffer::<BabyBear>::with_capacity(num_blocks).unwrap();

    //         input_device.stream().synchronize().unwrap();
    //         let time = Instant::now();
    //         unsafe {
    //             partial_sums.set_max_len();
    //             crate::baby_bear::ffi::partial_sum_baby_bear(
    //                 input_device.as_ptr() as *mut BabyBear,
    //                 partial_sums.as_mut_ptr(),
    //                 input_device.len(),
    //                 input_device.stream().handle(),
    //             )
    //         }
    //         input_device.stream().synchronize().unwrap();
    //         let elapsed = time.elapsed();
    //         println!("Device time: {:?}", elapsed);

    //         let partial_sums_from_device = partial_sums.to_host();

    //         let time = Instant::now();
    //         let partial_sums = input_host
    //             .chunks(512)
    //             .map(|chunk| chunk.iter().copied().sum::<BabyBear>())
    //             .collect::<Vec<_>>();
    //         let elapsed = time.elapsed();
    //         println!("host time: {:?}", elapsed);

    //         for (i, (dev_val, expected_val)) in
    //             partial_sums_from_device.into_iter().zip(partial_sums).enumerate()
    //         {
    //             assert_eq!(dev_val, expected_val, "failed at index {}", i);
    //         }

    //         println!("------------------------");
    //     }
    // }
}
