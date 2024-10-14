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
        pub fn add_baby_bear_vecs(
            a: *const F,
            b: *const F,
            c: *mut F,
            n: usize,
            stream: CudaStreamHandle,
        );
        pub fn compute_eq_poly(
            a: *mut F,
            c: *const F,
            n_low: usize,
            n_high: usize,
            stream: CudaStreamHandle,
        );

        pub fn compute_extension_eq_poly(
            a: *mut EF,
            c: *const EF,
            n_low: usize,
            n_high: usize,
            stream: CudaStreamHandle,
        );

        pub fn hadamard_product(
            a: *const F,
            b: *const F,
            c: *mut F,
            n_low: usize,
            n_high: usize,
            stream: CudaStreamHandle,
        );

        pub fn ef_hadamard_product(
            a: *const F,
            b: *const EF,
            c: *mut EF,
            n_low: usize,
            n_high: usize,
            stream: CudaStreamHandle,
        );

        pub fn sum_baby_bear_vec(a: *const F, c: *mut F, n: usize, stream: CudaStreamHandle);

        pub fn sum_baby_bear_vec_challenge(
            a: *const EF,
            c: *mut EF,
            n: usize,
            stream: CudaStreamHandle,
        );

        pub fn extension_multilinear_evaluator(
            result: *mut EF,
            point: *const EF,
            input_d: *mut EF,
            n_low: usize,
            n_high: usize,
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
    use spl_multi_pcs::{partial_lagrange_eval, Mle, Point};

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
    fn test_baby_bear_eq() {
        let n = 28;
        let num_repetitions = 1;
        let mut output = DeviceBuffer::<F>::with_capacity(1 << n).unwrap();
        let mut rng = thread_rng();
        for n_high in 0..10 {
            let n_low = n - n_high;
            for _ in 0..num_repetitions {
                let point = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();
                let point_d = point.to_device().unwrap();
                output.stream().synchronize().unwrap();
                let now = std::time::Instant::now();

                unsafe {
                    output.set_len(1 << n);
                    ffi::compute_eq_poly(
                        output.as_mut_ptr(),
                        point_d.as_ptr(),
                        n_low,
                        n_high,
                        point_d.stream().handle(),
                    );
                }

                output.stream().synchronize().unwrap();
                let elapsed = now.elapsed();
                println!("Cuda eq with n_high = {}  took {:?}", n_high, elapsed);
                let output = output.to_host();

                let now = std::time::Instant::now();
                let expected = partial_lagrange_eval(&Point::new(point));
                let elapsed = now.elapsed();
                println!("Sequential eq took {:?}", elapsed);

                for (elem_1, elem_2) in expected.iter().zip(output.iter()) {
                    assert_eq!(elem_1, elem_2);
                }
            }
        }
    }

    #[test]
    fn test_baby_bear_extension_eq() {
        let n = 28;
        let num_repetitions = 1;
        let mut output = DeviceBuffer::<EF>::with_capacity(1 << n).unwrap();
        let mut rng = thread_rng();
        for n_high in 0..10 {
            let n_low = n - n_high;
            for _ in 0..num_repetitions {
                let point = (0..n).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();
                let point_d = point.to_device().unwrap();
                output.stream().synchronize().unwrap();
                let now = std::time::Instant::now();

                unsafe {
                    output.set_len(1 << n);
                    ffi::compute_extension_eq_poly(
                        output.as_mut_ptr(),
                        point_d.as_ptr(),
                        n_low,
                        n_high,
                        point_d.stream().handle(),
                    );
                }

                output.stream().synchronize().unwrap();
                let elapsed = now.elapsed();
                println!("Cuda eq with n_high = {}  took {:?}", n_high, elapsed);
                let output = output.to_host();

                let now = std::time::Instant::now();
                let expected = spl_multi_pcs::partial_lagrange_eval(&Point::new(point));
                let elapsed = now.elapsed();
                println!("Sequential eq took {:?}", elapsed);

                for (elem_1, elem_2) in expected.iter().zip(output.iter()) {
                    assert_eq!(elem_1, elem_2);
                }
            }
        }
    }

    #[test]
    fn test_hadamard_product() {
        for power in 18..28 {
            let n = 1 << power;
            let mut rng = thread_rng();
            let a = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();
            let b = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();
            let c = (0..n).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();
            let a_d = a.to_device().unwrap();
            let b_d = b.to_device().unwrap();
            let c_d = c.to_device().unwrap();

            let mut d = DeviceBuffer::<F>::with_capacity(n).unwrap();

            let mut e = DeviceBuffer::<EF>::with_capacity(n).unwrap();

            unsafe {
                d.set_max_len();
                ffi::hadamard_product(
                    a_d.as_ptr(),
                    b_d.as_ptr(),
                    d.as_mut_ptr(),
                    power - 4,
                    4,
                    a_d.stream().handle(),
                );
                e.set_max_len();
                ffi::ef_hadamard_product(
                    a_d.as_ptr(),
                    c_d.as_ptr(),
                    e.as_mut_ptr(),
                    power - 4,
                    4,
                    a_d.stream().handle(),
                );
            }

            let d = d.to_host();
            let e = e.to_host();
            for (i, ((((a, b), c), d), e)) in a.into_iter().zip(b).zip(c).zip(d).zip(e).enumerate()
            {
                assert_eq!(a * b, d, "at index {}", i);
                assert_eq!(c * a, e, "at index {}", i);
            }
        }
    }

    #[test]
    fn test_sum_vec() {
        for power in 10..29 {
            let n = 1 << power;
            let mut rng = thread_rng();
            let a = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

            let mut a_d = a.to_device().unwrap();
            a_d.stream().synchronize().unwrap();
            let now = std::time::Instant::now();
            let mut result: Vec<F> = vec![F::zero(); 512];

            unsafe {
                ffi::sum_baby_bear_vec(a_d.as_ptr(), result.as_mut_ptr(), n, a_d.stream().handle());
            }

            let out_sum = result.into_iter().sum();
            a_d.stream().synchronize().unwrap();
            let elapsed = now.elapsed();

            drop(a_d);

            println!("Cuda sum took  for log height {} took {:?}", power, elapsed);

            let now = std::time::Instant::now();
            let sum: F = a.into_iter().sum();
            let elapsed = now.elapsed();
            println!("Sequential sum for log_height {} took {:?}", power, elapsed);

            assert_eq!(sum, out_sum);
        }
    }

    #[test]
    fn test_sum_extension_vec() {
        for power in 9..30 {
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

            let elapsed = now.elapsed();
            drop(a_d);
            println!("Cuda sum took  for log height {} took {:?}", power, elapsed);

            let now = std::time::Instant::now();
            let sum: EF = a.into_iter().sum();
            let elapsed = now.elapsed();
            println!("Sequential sum for log_height {} took {:?}", power, elapsed);

            assert_eq!(sum, out_sum);
        }
    }

    #[test]
    fn test_multilinear_eval() {
        for power in 10..29 {
            let n = 1 << power;
            let mut rng = thread_rng();
            let a = (0..n).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();
            let point = (0..power).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();

            let mut a_d = a.to_device().unwrap();
            let point_d = point.to_device().unwrap();
            a_d.stream().synchronize().unwrap();
            let mut result: Vec<EF> = vec![EF::zero(); 512];
            let now = std::time::Instant::now();

            unsafe {
                ffi::extension_multilinear_evaluator(
                    result.as_mut_ptr(),
                    point_d.as_ptr(),
                    a_d.as_mut_ptr(),
                    power - 4,
                    4,
                    a_d.stream().handle(),
                );
            }

            let out_sum: EF = result.into_iter().sum();
            a_d.stream().synchronize().unwrap();
            let elapsed = now.elapsed();
            println!("Cuda sum for log height {} took {:?}", power, elapsed);

            let now = std::time::Instant::now();
            let eval: EF = Mle::eval_at_point(&a.into(), &Point::new(point));
            let elapsed = now.elapsed();
            println!("Sequential sum for log_height {} took {:?}", power, elapsed);

            assert_eq!(out_sum, eval);
        }
    }
}
