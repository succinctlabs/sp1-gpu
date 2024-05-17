use p3_baby_bear::BabyBear;

use crate::device::slice::DeviceSlice;

mod ffi;

#[derive(Debug, Clone, Copy)]
pub struct DeviceDft;

impl DeviceDft {
    /// # Safety
    pub unsafe fn coset_lde(
        &self,
        inout_slice: &mut DeviceSlice<BabyBear>,
        log_degree: usize,
        log_blowup: usize,
    ) {
        unsafe { ffi::coset_lde(inout_slice.as_mut_ptr(), log_degree, log_blowup) };
    }

    /// # Safety
    pub unsafe fn dft(&self, inout_slice: &mut DeviceSlice<BabyBear>, log_degree: usize) {
        unsafe { ffi::dft(inout_slice.as_mut_ptr(), log_degree) };
    }

    /// # Safety
    pub unsafe fn idft(&self, inout_slice: &mut DeviceSlice<BabyBear>, log_degree: usize) {
        unsafe { ffi::idft(inout_slice.as_mut_ptr(), log_degree) };
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use p3_baby_bear::BabyBear;
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use rand::{thread_rng, Rng};

    use super::DeviceDft;
    use crate::device::buffer::{DeviceBuffer, ToDevice};
    use p3_field::AbstractField;

    #[test]
    fn test_dft() {
        let mut rng = thread_rng();

        let log_degrees = 1..28;

        let dft = DeviceDft;
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees {
            if log_d == 16 || log_d == 17 || log_d == 18 || log_d == 19 {
                continue;
            }
            let d = 1 << log_d;
            let values = (0..d).map(|_| rng.gen()).collect::<Vec<BabyBear>>();

            let mut d_values = values.clone().to_device();

            let time = Instant::now();
            unsafe { dft.dft(&mut d_values[..], log_d) };
            let gpu_time = time.elapsed();
            println!("Gpu dft time log degree {}: {:?}", log_d, gpu_time);

            let time = Instant::now();
            let expected_value = p3_dft.dft(values);
            let cpu_time = time.elapsed();
            println!("Cpu dft time log degree {}: {:?}", log_d, cpu_time);

            let mut values_back = vec![BabyBear::zero(); d];
            d_values.copy_to_host(&mut values_back);

            for (val, exp) in values_back.into_iter().zip(expected_value) {
                assert_eq!(val, exp);
            }
        }
    }

    #[test]
    fn test_idft() {
        let mut rng = thread_rng();

        let log_degrees = 1..28;

        let dft = DeviceDft;
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees {
            if log_d == 16 || log_d == 17 || log_d == 18 || log_d == 19 {
                continue;
            }
            let d = 1 << log_d;
            let values = (0..d).map(|_| rng.gen()).collect::<Vec<BabyBear>>();

            let mut d_values = values.clone().to_device();

            let time = Instant::now();
            unsafe { dft.idft(&mut d_values[..], log_d) };
            let gpu_time = time.elapsed();
            println!("Gpu idft time log degree {}: {:?}", log_d, gpu_time);

            let time = Instant::now();
            let expected_value = p3_dft.idft(values);
            let cpu_time = time.elapsed();
            println!("Cpu idft time log degree {}: {:?}", log_d, cpu_time);

            let mut values_back = vec![BabyBear::zero(); d];
            d_values.copy_to_host(&mut values_back);

            for (val, exp) in values_back.into_iter().zip(expected_value) {
                assert_eq!(val, exp);
            }
        }
    }

    #[test]
    fn test_coset_lde() {
        let mut rng = thread_rng();

        let log_degrees = 20..24;
        let log_blowup = 2;

        let dft = DeviceDft;
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees {
            let d = 1 << log_d;
            let ext_d = d << log_blowup;

            let values = (0..d).map(|_| rng.gen()).collect::<Vec<BabyBear>>();

            let mut d_values = DeviceBuffer::<BabyBear>::new(ext_d);
            d_values.extend_from_host_slice(&vec![BabyBear::zero(); ext_d - d]);
            d_values.extend_from_host_slice(&values);

            let time = Instant::now();
            unsafe { dft.coset_lde(&mut d_values[..], log_d, log_blowup) };
            let gpu_time = time.elapsed();
            println!("Gpu lde time log degree {}: {:?}", log_d, gpu_time);

            let time = Instant::now();
            let expected_value = p3_dft.coset_lde(values, log_blowup, BabyBear::generator());
            let cpu_time = time.elapsed();
            println!("Cpu lde time log degree {}: {:?}", log_d, cpu_time);

            let mut values_back = vec![BabyBear::zero(); ext_d];
            d_values.copy_to_host(&mut values_back);

            for (val, exp) in values_back.into_iter().zip(expected_value) {
                assert_eq!(val, exp);
            }
        }
    }
}
