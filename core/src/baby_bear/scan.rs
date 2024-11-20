use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;

use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaRustError, CudaScan},
};

use super::ffi;

type F = BabyBear;
pub const D: usize = 4;
type EF = BinomialExtensionField<F, D>;

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

    use crate::device::memory::{ToDevice, ToHost};

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
}
