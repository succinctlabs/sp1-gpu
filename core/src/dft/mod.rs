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
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use rand::{thread_rng, Rng};

    use super::DeviceDft;
    use crate::device::buffer::ToDevice;
    use p3_field::AbstractField;

    #[test]
    fn test_dft() {
        let mut rng = thread_rng();

        let log_degrees = [10, 11, 12, 13];

        let dft = DeviceDft;
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees {
            let d = 1 << log_d;
            let values = (0..d).map(|_| rng.gen()).collect::<Vec<BabyBear>>();

            let mut d_values = values.clone().to_device();
            unsafe { dft.dft(&mut d_values[..], log_d) };

            let expected_value = p3_dft.dft(values);

            let mut values_back = vec![BabyBear::zero(); d];
            d_values.copy_to_host(&mut values_back);

            assert_eq!(expected_value.len(), values_back.len());

            for (val, exp) in values_back.into_iter().zip(expected_value) {
                assert_eq!(val, exp);
            }
        }
    }
}
