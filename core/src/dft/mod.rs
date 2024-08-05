use std::marker::PhantomData;

use p3_baby_bear::BabyBear;

use crate::{
    device::{error::CudaError, DevicePointer, DeviceStreamPointer, RawPointer},
    matrix::ColMajorMatrix,
};

mod ffi;

pub trait Dft<P: RawPointer> {
    fn coset_lde_batch(
        &self,
        matrix: &mut ColMajorMatrix<P>,
        log_blowup: usize,
        shift: BabyBear,
        bit_rev: bool,
    ) -> Result<(), CudaError>;
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceDft<F>(PhantomData<F>);

impl Dft<DevicePointer<BabyBear>> for DeviceDft<BabyBear> {
    fn coset_lde_batch(
        &self,
        matrix: &mut ColMajorMatrix<DevicePointer<BabyBear>>,
        log_blowup: usize,
        shift: BabyBear,
        bit_rev: bool,
    ) -> Result<(), CudaError> {
        unsafe {
            ffi::batch_lde_shift(
                matrix.values.as_mut_ptr(),
                matrix.height.ilog2() - log_blowup as u32,
                log_blowup as u32,
                shift,
                matrix.width() as u32,
                bit_rev,
            )
            .into()
        }
    }
}

impl Dft<DeviceStreamPointer<BabyBear>> for DeviceDft<BabyBear> {
    fn coset_lde_batch(
        &self,
        matrix: &mut ColMajorMatrix<DeviceStreamPointer<BabyBear>>,
        log_blowup: usize,
        shift: BabyBear,
        bit_rev: bool,
    ) -> Result<(), CudaError> {
        matrix.values.stream().synchronize();
        unsafe {
            ffi::batch_lde_shift(
                matrix.values.as_mut_ptr(),
                matrix.height.ilog2() - log_blowup as u32,
                log_blowup as u32,
                shift,
                matrix.width() as u32,
                bit_rev,
            )
            .into()
        }
    }
}

impl DeviceDft<BabyBear> {
    pub fn init() -> Result<Self, CudaError> {
        Result::from(unsafe { ffi::sppark_init() })?;
        Ok(Self(PhantomData))
    }

    pub fn new() -> Self {
        Self::init().unwrap()
    }
}

impl Default for DeviceDft<BabyBear> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::dft::Dft;
    use std::time::Instant;

    use p3_baby_bear::BabyBear;
    use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use p3_matrix::{bitrev::BitReversableMatrix, dense::RowMajorMatrix, Matrix};
    use rand::{thread_rng, Rng};

    use super::DeviceDft;
    use crate::device::memory::{ToDevice, ToHost};
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
    fn test_batch_coset_lde() {
        let mut rng = thread_rng();

        let log_degrees = 16..18;
        let log_blowup = 1;
        let batch_size = 100;

        let dft = DeviceDft::new();
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees.clone() {
            let d = 1 << log_d;

            let mat_h = RowMajorMatrix::rand(&mut rng, d, batch_size);
            let mut mat_d = unsafe {
                mat_h
                    .to_device()
                    .unwrap()
                    .embed_as_blowup(log_blowup)
                    .unwrap()
            };

            // Test the regulat version.
            let time = Instant::now();
            let shift = rng.gen::<BabyBear>();
            dft.coset_lde_batch(&mut mat_d, log_blowup, shift, false)
                .unwrap();
            let gpu_time = time.elapsed();
            println!("Gpu lde time log degree {}: {:?}", log_d, gpu_time);

            let mat_h_clone = mat_h.clone();
            let time = Instant::now();
            let expected_value = p3_dft
                .coset_lde_batch(mat_h_clone, log_blowup, BabyBear::generator() * shift)
                .to_row_major_matrix();
            let cpu_time = time.elapsed();
            println!("Cpu lde time log degree {}: {:?}", log_d, cpu_time);

            let values_back = mat_d.to_host();

            for (val, exp) in values_back.values.into_iter().zip(expected_value.values) {
                assert_eq!(val, exp);
            }
        }
    }

    #[test]
    fn test_batch_bit_reversed_coset_lde() {
        let mut rng = thread_rng();

        let log_degrees = 4..20;
        let log_blowup = 1;
        let batch_size = 100;

        let dft = DeviceDft::new();
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees.clone() {
            let d = 1 << log_d;

            let mat_h = RowMajorMatrix::rand(&mut rng, d, batch_size);
            let mut mat_d = unsafe {
                mat_h
                    .to_device()
                    .unwrap()
                    .embed_as_blowup(log_blowup)
                    .unwrap()
            };

            // Test the regulat version.
            let time = Instant::now();
            dft.coset_lde_batch(&mut mat_d, log_blowup, BabyBear::one(), true)
                .unwrap();
            let gpu_time = time.elapsed();
            println!("Gpu lde time log degree {}: {:?}", log_d, gpu_time);

            let mat_h_clone = mat_h.clone();
            let time = Instant::now();
            let expected_value = p3_dft
                .coset_lde_batch(mat_h_clone, log_blowup, BabyBear::generator())
                .bit_reverse_rows()
                .to_row_major_matrix();
            let cpu_time = time.elapsed();
            println!("Cpu lde time log degree {}: {:?}", log_d, cpu_time);

            let values_back = mat_d.to_host();

            for (val, exp) in values_back.values.into_iter().zip(expected_value.values) {
                assert_eq!(val, exp);
            }
        }
    }
}
