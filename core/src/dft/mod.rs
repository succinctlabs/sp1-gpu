use std::marker::PhantomData;

use p3_baby_bear::BabyBear;

use crate::{
    device::{error::CudaError, slice::DeviceSlice},
    matrix::MatrixViewMutDevice,
};

mod ffi;

#[derive(Debug, Clone, Copy)]
pub struct DeviceDft<F = BabyBear>(PhantomData<F>);

impl DeviceDft<BabyBear> {
    pub fn init() -> Result<Self, CudaError> {
        Result::from(unsafe { ffi::sppark_init() })?;
        Ok(Self(PhantomData))
    }

    pub fn new() -> Self {
        Self::init().unwrap()
    }

    /// # Safety
    pub unsafe fn dft_device(
        &self,
        inout_slice: &mut DeviceSlice<BabyBear>,
        log_degree: usize,
    ) -> Result<(), CudaError> {
        ffi::batch_NTT(inout_slice.as_mut_ptr(), log_degree as u32, 1).into()
    }

    /// # Safety
    pub unsafe fn dft_batch_device(
        &self,
        matrix: MatrixViewMutDevice<BabyBear>,
    ) -> Result<(), CudaError> {
        assert!(!matrix.row_major);
        ffi::batch_NTT(matrix.values, matrix.height.ilog2(), matrix.width as u32).into()
    }

    /// # Safety
    pub unsafe fn idft_device(
        &self,
        inout_slice: &mut DeviceSlice<BabyBear>,
        log_degree: usize,
    ) -> Result<(), CudaError> {
        ffi::batch_iNTT(inout_slice.as_mut_ptr(), log_degree as u32, 1).into()
    }

    /// # Safety
    pub unsafe fn idft_batch_device(
        &self,
        matrix: MatrixViewMutDevice<BabyBear>,
    ) -> Result<(), CudaError> {
        assert!(!matrix.row_major);
        ffi::batch_iNTT(matrix.values, matrix.height.ilog2(), matrix.width as u32).into()
    }

    /// # Safety
    pub unsafe fn coset_lde_device(
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
            false,
        )
        .into()
    }

    /// # Safety
    pub unsafe fn coset_lde_batch_device(
        &self,
        matrix: MatrixViewMutDevice<BabyBear>,
        log_blowup: usize,
        bit_rev: bool,
    ) -> Result<(), CudaError> {
        assert!(!matrix.row_major);
        ffi::batch_lde_shift(
            matrix.values,
            matrix.height.ilog2() - log_blowup as u32,
            log_blowup as u32,
            matrix.width as u32,
            bit_rev,
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
    use p3_matrix::{bitrev::BitReversableMatrix, dense::RowMajorMatrix, Matrix};
    use rand::{thread_rng, Rng};

    use super::DeviceDft;
    use crate::{
        device::{
            buffer::DeviceBuffer,
            memory::{ToDevice, ToHost},
        },
        matrix::ColMajorMatrixDevice,
    };
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
    fn test_batch_dft() {
        let log_degrees = 10..20;
        let batch_size = 100;

        let dft = DeviceDft::new();
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees {
            let d = 1 << log_d;

            let (mat_h, mut mat_d) = ColMajorMatrixDevice::<BabyBear>::dummy(batch_size, d);

            assert_eq!(mat_d.width(), batch_size);
            assert_eq!(mat_d.height(), d);

            let time = Instant::now();
            unsafe { dft.dft_batch_device(mat_d.view_mut()) }.unwrap();
            let gpu_time = time.elapsed();
            println!("Gpu dft time log degree {}: {:?}", log_d, gpu_time);

            let time = Instant::now();
            let expected_value = p3_dft.dft_batch(mat_h).to_row_major_matrix();
            let cpu_time = time.elapsed();
            println!("Cpu dft time log degree {}: {:?}", log_d, cpu_time);

            let results = mat_d.to_host();
            assert_eq!(results.width(), batch_size);
            assert_eq!(results.height(), d);

            for (val, exp) in results.values.into_iter().zip(expected_value.values) {
                assert_eq!(val, exp);
            }
        }
    }

    #[test]
    fn test_dft() {
        let mut rng = thread_rng();

        let log_degrees = 10..20;

        let dft = DeviceDft::new();
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees {
            let d = 1 << log_d;
            let values = (0..d).map(|_| rng.gen()).collect::<Vec<BabyBear>>();

            let mut d_values = DeviceBuffer::with_capacity(d);
            d_values.extend_from_host_slice(&values);

            let time = Instant::now();
            unsafe { dft.dft_device(&mut d_values[..], log_d) }.unwrap();
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

        let log_degrees = 10..20;

        let dft = DeviceDft::new();
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees {
            let d = 1 << log_d;
            let values = (0..d).map(|_| rng.gen()).collect::<Vec<BabyBear>>();

            let mut d_values = values.clone().to_device();

            let time = Instant::now();
            unsafe { dft.idft_device(&mut d_values[..], log_d) }.unwrap();
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

        let log_degrees = 4..20;
        let log_blowup = 2;

        let dft = DeviceDft::new();
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees.clone() {
            let d = 1 << log_d;
            let ext_d = d << log_blowup;

            let mut d_values = DeviceBuffer::<BabyBear>::with_capacity(ext_d);

            let values = (0..d).map(|_| rng.gen()).collect::<Vec<BabyBear>>();

            d_values.extend_from_host_slice(&vec![BabyBear::zero(); ext_d - d]);
            d_values.extend_from_host_slice(&values);

            let time = Instant::now();
            unsafe { dft.coset_lde_device(&mut d_values[..], log_d, log_blowup) }.unwrap();
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

    #[test]
    fn test_batch_coset_lde() {
        let mut rng = thread_rng();

        let log_degrees = 4..20;
        let log_blowup = 1;
        let batch_size = 100;

        let dft = DeviceDft::new();
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees.clone() {
            let d = 1 << log_d;

            let mat_h = RowMajorMatrix::rand(&mut rng, d, batch_size);
            let mut mat_d = mat_h.to_device().to_column_major_blowup(log_blowup);

            // Test the regulat version.
            let time = Instant::now();
            unsafe { dft.coset_lde_batch_device(mat_d.view_mut(), log_blowup, false) }.unwrap();
            let gpu_time = time.elapsed();
            println!("Gpu lde time log degree {}: {:?}", log_d, gpu_time);

            let mat_h_clone = mat_h.clone();
            let time = Instant::now();
            let expected_value = p3_dft
                .coset_lde_batch(mat_h_clone, log_blowup, BabyBear::generator())
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
            let mut mat_d = mat_h.to_device().to_column_major_blowup(log_blowup);

            // Test the regulat version.
            let time = Instant::now();
            unsafe { dft.coset_lde_batch_device(mat_d.view_mut(), log_blowup, true) }.unwrap();
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
