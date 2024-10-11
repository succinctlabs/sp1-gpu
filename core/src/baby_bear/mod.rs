use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaRustError, CudaScan},
};

type F = BabyBear;
const D: usize = 4;
type EF = BinomialExtensionField<F, D>;

mod ffi {
    use super::*;
    use crate::{cuda_runtime::stream::CudaStreamHandle, device::error::CudaRustError};

    extern "C" {
        pub fn scan_baby_bear(
            a: *const F,
            b: *const F,
            n: usize,
            stream: CudaStreamHandle,
        ) -> CudaRustError;
        pub fn scan_baby_bear_challenge(
            a: *const EF,
            b: *const EF,
            n: usize,
            stream: CudaStreamHandle,
        ) -> CudaRustError;
        pub(crate) fn add_baby_bear_vecs(
            a: *const F,
            b: *const F,
            c: *mut F,
            n: usize,
            stream: CudaStreamHandle,
        );
        pub(crate) fn sum_baby_bear_vec(
            a: *const F,
            out: *mut F,
            n: usize,
            stream: CudaStreamHandle,
        );
        pub(crate) fn sum_baby_bear_vec_challenge(
            a: *const EF,
            out: *mut EF,
            n: usize,
            stream: CudaStreamHandle,
        );
    }
}

impl CudaScan for F {
    unsafe fn cuda_scan(
        a: *mut Self,
        b: *const Self,
        n: usize,
        stream: &CudaStream,
    ) -> CudaRustError {
        ffi::scan_baby_bear(a, b, n, stream.handle())
    }
}

impl CudaScan for EF {
    unsafe fn cuda_scan(
        a: *mut Self,
        b: *const Self,
        n: usize,
        stream: &CudaStream,
    ) -> CudaRustError {
        ffi::scan_baby_bear_challenge(a, b, n, stream.handle())
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use rand::{thread_rng, Rng};
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    use crate::device::{
        memory::{ToDevice, ToHost},
        DeviceBuffer,
    };

    use super::*;

    #[test]
    fn test_baby_bear_scan() {
        let small_n: usize = 250;
        let large_n: usize = 1 << 22;

        let mut rng = thread_rng();

        for n in [small_n, large_n] {
            let a_h = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();
            let a = a_h.to_device().unwrap();

            let res = a.scan().unwrap();
            let res_h = res.to_host();

            for (i, (exp, res)) in a_h
                .into_iter()
                .scan(F::zero(), |acc, x| {
                    *acc += x;
                    Some(*acc)
                })
                .zip(res_h)
                .enumerate()
            {
                assert_eq!(exp, res, "at index {}", i);
            }
        }
    }

    #[test]
    fn test_baby_bear_challenge_scan() {
        let small_n: usize = 250;
        let large_n: usize = 1 << 22;

        let mut rng = thread_rng();

        for n in [small_n, large_n] {
            let a_h = (0..n).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();
            let mut a = a_h.to_device().unwrap();

            // let res = a.scan().unwrap();
            a.scan_inplace().unwrap();
            let res = a;
            let res_h = res.to_host();

            for (i, (exp, res)) in a_h
                .into_iter()
                .scan(EF::zero(), |acc, x| {
                    *acc += x;
                    Some(*acc)
                })
                .zip(res_h)
                .enumerate()
            {
                assert_eq!(exp, res, "at index {}", i);
            }
        }
    }

    #[test]
    fn test_add_vecs() {
        let n = 1 << 25;
        let mut rng = thread_rng();
        let a = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();
        let b = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();
        let a_d = a.to_device().unwrap();
        let b_d = b.to_device().unwrap();

        let mut c = DeviceBuffer::<F>::with_capacity(n).unwrap();

        unsafe {
            c.set_max_len();
            ffi::add_baby_bear_vecs(
                a_d.as_ptr(),
                b_d.as_ptr(),
                c.as_mut_ptr(),
                n,
                a_d.stream().handle(),
            );
        }

        let c = c.to_host();
        for (i, ((a, b), c)) in a.into_iter().zip(b).zip(c).enumerate() {
            assert_eq!(a + b, c, "at index {}", i);
        }
    }

    #[test]
    fn test_sum_vec() {
        for power in 10..27 {
            let n = 1 << power;
            let mut rng = thread_rng();
            let a = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

            let a_d = a.to_device().unwrap();
            a_d.stream().synchronize().unwrap();
            let now = std::time::Instant::now();
            let mut result: Vec<F> = vec![F::zero(); 512];

            unsafe {
                ffi::sum_baby_bear_vec(a_d.as_ptr(), result.as_mut_ptr(), n, a_d.stream().handle());
            }

            let out_sum = result.into_iter().sum();
            a_d.stream().synchronize().unwrap();
            let elapsed = now.elapsed();
            println!("Cuda sum took  for log height {} took {:?}", power, elapsed);

            let now = std::time::Instant::now();
            let sum: F = a.clone().into_par_iter().sum();
            let elapsed = now.elapsed();
            println!("Sequential sum for log_height {} took {:?}", power, elapsed);

            assert_eq!(sum, out_sum);
        }
    }

    #[test]
    fn test_sum_extension_vec() {
        for power in 10..27 {
            let n = 1 << power;
            let mut rng = thread_rng();
            let a = (0..n).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();

            let a_d = a.to_device().unwrap();
            a_d.stream().synchronize().unwrap();
            let now = std::time::Instant::now();
            let mut result: Vec<EF> = vec![EF::zero(); 512];

            unsafe {
                ffi::sum_baby_bear_vec_challenge(
                    a_d.as_ptr(),
                    result.as_mut_ptr(),
                    n,
                    a_d.stream().handle(),
                );
            }

            let out_sum = result.into_iter().sum();
            a_d.stream().synchronize().unwrap();
            let elapsed = now.elapsed();
            println!("Cuda sum took  for log height {} took {:?}", power, elapsed);

            let now = std::time::Instant::now();
            let sum: EF = a.clone().into_par_iter().sum();
            let elapsed = now.elapsed();
            println!("Sequential sum for log_height {} took {:?}", power, elapsed);

            assert_eq!(sum, out_sum);
        }
    }
}
