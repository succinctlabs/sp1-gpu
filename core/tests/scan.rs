use moongate_core::device::buffer::DeviceBuffer;
use moongate_core::device::buffer::ToDevice;
use moongate_core::device::error::CudaRustError;
use moongate_core::time::CudaInstant;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::thread_rng;
use rand::Rng;

type F = BabyBear;

extern "C" {
    pub fn scan_baby_bear(a: *const F, b: *const F, n: usize) -> CudaRustError;
}

#[test]
fn test_device_small_scan() {
    let n: usize = 200;

    let mut rng = thread_rng();

    let a_h = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();
    let a = a_h.to_device();

    let mut res = DeviceBuffer::<F>::with_capacity(n);
    unsafe {
        res.set_max_len();
        scan_baby_bear(a.as_ptr(), res.as_mut_ptr(), n)
            .to_result()
            .unwrap();
    }

    let res_h = res.to_host();
    for (exp, res) in a_h
        .into_iter()
        .scan(F::zero(), |acc, x| {
            *acc += x;
            Some(*acc)
        })
        .zip(res_h)
    {
        assert_eq!(exp, res);
    }
}

#[test]
fn test_device_large_scan() {
    let n: usize = 1 << 22;

    let mut rng = thread_rng();

    let a_h = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();
    let a = a_h.to_device();

    let mut res = DeviceBuffer::<F>::with_capacity(n);
    let time = CudaInstant::now().unwrap();
    unsafe {
        res.set_max_len();
        scan_baby_bear(a.as_ptr(), res.as_mut_ptr(), n)
            .to_result()
            .unwrap();
    }
    println!("scan time: {:?}", time.elapsed().unwrap());

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
