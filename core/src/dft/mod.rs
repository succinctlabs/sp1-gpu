use std::marker::PhantomData;

use p3_baby_bear::BabyBear;

use crate::device::{error::CudaError, slice::DeviceSlice};

mod ffi;

#[derive(Debug, Clone, Copy)]
pub struct DeviceDft(PhantomData<()>);

impl DeviceDft {
    pub fn init() -> Result<Self, CudaError> {
        Result::from(unsafe { ffi::sppark_init() })?;
        Ok(Self(PhantomData))
    }

    pub fn new() -> Self {
        Self::init().unwrap()
    }

    /// # Safety
    pub unsafe fn dft(
        &self,
        inout_slice: &mut DeviceSlice<BabyBear>,
        log_degree: usize,
    ) -> Result<(), CudaError> {
        ffi::batch_NTT(inout_slice.as_mut_ptr(), log_degree as u32, 1).into()
    }

    /// # Safety
    pub unsafe fn idft(
        &self,
        inout_slice: &mut DeviceSlice<BabyBear>,
        log_degree: usize,
    ) -> Result<(), CudaError> {
        ffi::batch_iNTT(inout_slice.as_mut_ptr(), log_degree as u32, 1).into()
    }

    /// # Safety
    pub unsafe fn coset_lde(
        &self,
        inout_slice: &mut DeviceSlice<BabyBear>,
        log_degree: usize,
        log_blowup: usize,
    ) -> Result<(), CudaError> {
        ffi::batch_lde_shift(
            inout_slice.as_mut_ptr(),
            log_degree as u32,
            log_blowup as u32,
            1,
        )
        .into()
    }
}

impl Default for DeviceDft {
    fn default() -> Self {
        Self::new()
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
    use p3_field::{AbstractField, Field, PrimeField32, TwoAdicField};

    #[test]
    fn make_roots() {
        /// The Baby Bear prime
        /// This is the unique 31-bit prime with the highest possible 2 adicity (27).
        const P: u32 = 0x78000001;
        const MONTY_BITS: u32 = 32;

        #[inline]
        #[must_use]
        const fn to_monty(x: u32) -> u32 {
            (((x as u64) << MONTY_BITS) % P as u64) as u32
        }

        assert_eq!(
            0xfffffbe,
            to_monty(BabyBear::generator().as_canonical_u32())
        );

        assert_eq!(
            0x17bdef7c,
            to_monty(BabyBear::generator().inverse().as_canonical_u32())
        );

        for i in 0..28 {
            println!(
                "    fr_t({:x}u),",
                to_monty(BabyBear::two_adic_generator(i).as_canonical_u32())
            );
        }

        println!("Inverses: ");

        for i in 0..28 {
            println!(
                "    fr_t({:x}u),",
                to_monty(BabyBear::two_adic_generator(i).inverse().as_canonical_u32())
            );
        }

        let mut root = BabyBear::two_adic_generator(27);
        for i in (0..27).rev() {
            let current_root = BabyBear::two_adic_generator(i);
            assert_eq!(root.square(), current_root);
            root = current_root;
        }
    }

    #[test]
    fn test_dft() {
        let mut rng = thread_rng();

        let log_degrees = 1..28;

        let dft = DeviceDft::new();
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees {
            let d = 1 << log_d;
            let values = (0..d).map(|_| rng.gen()).collect::<Vec<BabyBear>>();

            let mut d_values = DeviceBuffer::new(d);
            d_values.extend_from_host_slice(&values);

            let time = Instant::now();
            unsafe { dft.dft(&mut d_values[..], log_d) }.unwrap();
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

        let dft = DeviceDft::new();
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees {
            let d = 1 << log_d;
            let values = (0..d).map(|_| rng.gen()).collect::<Vec<BabyBear>>();

            let mut d_values = values.clone().to_device();

            let time = Instant::now();
            unsafe { dft.idft(&mut d_values[..], log_d) }.unwrap();
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

        let log_degrees = 4..26;
        let log_blowup = 1;

        let dft = DeviceDft::new();
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees.clone() {
            let d = 1 << log_d;
            let ext_d = d << log_blowup;

            let mut d_values = DeviceBuffer::<BabyBear>::new(ext_d);

            let values = (0..d).map(|_| rng.gen()).collect::<Vec<BabyBear>>();

            d_values.extend_from_host_slice(&vec![BabyBear::zero(); ext_d - d]);
            d_values.extend_from_host_slice(&values);

            let time = Instant::now();
            unsafe { dft.coset_lde(&mut d_values[..], log_d, log_blowup) }.unwrap();
            let gpu_time = time.elapsed();
            println!("Gpu lde time log degree {}: {:?}", log_d, gpu_time);

            let time = Instant::now();
            let expected_value = p3_dft.coset_lde(values, log_blowup, BabyBear::generator());
            let cpu_time = time.elapsed();
            println!("Cpu lde time log degree {}: {:?}", log_d, cpu_time);

            let mut values_back = vec![BabyBear::zero(); ext_d];
            d_values[0..ext_d].copy_into_host(&mut values_back);

            for (val, exp) in values_back.into_iter().zip(expected_value) {
                assert_eq!(val, exp);
            }
        }
    }
}
