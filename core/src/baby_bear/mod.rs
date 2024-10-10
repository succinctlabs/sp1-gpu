use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField, Field};

use rayon::iter::{IntoParallelIterator, ParallelIterator};

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
        pub fn add_baby_bear_vecs(
            a: *const F,
            b: *const F,
            c: *mut F,
            n: usize,
            stream: CudaStreamHandle,
        );
        pub fn sum_baby_bear_vec(a: *const F, out: *mut F, n: usize, stream: CudaStreamHandle);
        pub fn compute_eq_poly(a: *mut F, c: *const F, n: usize, stream: CudaStreamHandle);
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
    use p3_util::log2_ceil_usize;
    use rand::{thread_rng, Rng};
    use spl_multi_pcs::Point;

    use crate::{
        cuda_runtime::ffi::cuda_stream_synchronize,
        device::{
            memory::{ToDevice, ToHost},
            DeviceBuffer,
        },
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
        let mut a_d = a.to_device().unwrap();
        let mut b_d = b.to_device().unwrap();

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
        for power in (10..32) {
            let n = 1 << power;
            let mut rng = thread_rng();
            let a = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

            let num_rounds = log2_ceil_usize(n) / 9;

            println!("Num rounds: {}", num_rounds);

            let now = std::time::Instant::now();
            let mut a_d = a.to_device().unwrap();
            let mut result;
            result = DeviceBuffer::<F>::with_capacity(n.div_ceil(512)).unwrap();
            // let mut scratch = DeviceBuffer::<F>::with_capacity()
            unsafe {
                result.set_len(n.div_ceil(512));
            }
            let mut new_size = n;
            for _ in 0..num_rounds {
                unsafe {
                    ffi::sum_baby_bear_vec(
                        a_d.as_ptr(),
                        result.as_mut_ptr(),
                        new_size,
                        a_d.stream().handle(),
                    )
                };
                a_d = result;
                new_size = new_size.div_ceil(512);
                result = DeviceBuffer::<F>::with_capacity(new_size).unwrap();
                unsafe {
                    a_d.set_len(new_size);
                };
            }

            let result = a_d.to_host();
            let out_sum = result.into_iter().sum();
            println!("Cuda sum took {:?}", now.elapsed());

            let now = std::time::Instant::now();
            let sum: F = a.clone().into_par_iter().sum();
            println!("Sequential sum took {:?}", now.elapsed());

            assert_eq!(sum, out_sum);
        }
    }

    #[test]
    fn test_baby_bear_eq() {
        let n = 30;
        let mut rng = thread_rng();
        let point = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();
        let point_d = point.to_device().unwrap();
        let mut output = DeviceBuffer::<F>::with_capacity(1 << n).unwrap();
        let now = std::time::Instant::now();

        unsafe {
            output.set_len(1 << n);
            ffi::compute_eq_poly(
                output.as_mut_ptr(),
                point_d.as_ptr(),
                n,
                point_d.stream().handle(),
            );
        }

        println!("Cuda eq took {:?}", now.elapsed());
        let output = output.to_host();

        let now = std::time::Instant::now();
        let expected = spl_multi_pcs::partial_lagrange_eval(&Point::new(point).reversed_point());
        println!("Sequential eq took {:?}", now.elapsed());

        assert_eq!(expected, output);
    }
}
