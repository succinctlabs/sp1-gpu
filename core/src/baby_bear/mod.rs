use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

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
        let n: usize = 1 << 27;
        let mut rng = thread_rng();
        let a = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let num_rounds = log2_ceil_usize(n).div_ceil(9);

        println!("Num rounds: {}", num_rounds);

        let mut a_d = a.to_device().unwrap();
        let mut result;
        result = DeviceBuffer::<F>::with_capacity(n.div_ceil(512)).unwrap();
        let now = std::time::Instant::now();
        unsafe {
            result.set_max_len();
        }
        for i in 0..num_rounds {
            unsafe {
                ffi::sum_baby_bear_vec(a_d.as_ptr(), result.as_mut_ptr(), n, a_d.stream().handle())
            };
            a_d = result;
            result =
                DeviceBuffer::<F>::with_capacity(n.div_ceil(512_u32.pow(i as u32 + 1) as usize))
                    .unwrap();
            unsafe { result.set_max_len() };
        }

        println!("Cuda sum took {:?}", now.elapsed());
        let result = a_d.to_host();

        let now = std::time::Instant::now();
        let sum: F = a.clone().into_par_iter().sum();
        println!("Sequential sum took {:?}", now.elapsed());

        let out_sum: F = result[0];

        assert_eq!(sum, out_sum);
    }
}
