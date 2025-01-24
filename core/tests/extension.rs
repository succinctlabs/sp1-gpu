#![allow(unused_doc_comments)]

use itertools::izip;
use moongate_core::device::{
    memory::{ToDevice, ToHost},
    DeviceBuffer,
};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};
use rand::{thread_rng, Rng};

const D: usize = 4;
type F = BabyBear;
type EF = BinomialExtensionField<F, D>;

/// cbindgen:ignore
extern "C" {
    pub fn test_bb31_extension(
        a: *const EF,
        b: *const EF,
        add: *mut EF,
        sub: *mut EF,
        mul: *mut EF,
        div: *mut EF,
        n: usize,
        block_size: usize,
        grid_size: usize,
    );
}

#[test]
fn test_device_extension() {
    let n: usize = 10000;
    let block_size = 256;
    let grid_size = n.div_ceil(block_size);

    let mut rng = thread_rng();

    let mut a_h = (0..n - 2).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();
    a_h.push(EF::zero());
    a_h.push(EF::one());
    let mut b_h = (0..n - 1).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();
    b_h.push(EF::one());

    let a = a_h.to_device().unwrap();
    let b = b_h.to_device().unwrap();

    let mut add = DeviceBuffer::<EF>::with_capacity(n).unwrap();
    let mut sub = DeviceBuffer::<EF>::with_capacity(n).unwrap();
    let mut mul = DeviceBuffer::<EF>::with_capacity(n).unwrap();
    let mut div = DeviceBuffer::<EF>::with_capacity(n).unwrap();

    unsafe {
        add.set_max_len();
        sub.set_max_len();
        mul.set_max_len();
        div.set_max_len();
        test_bb31_extension(
            a.as_ptr(),
            b.as_ptr(),
            add.as_mut_ptr(),
            sub.as_mut_ptr(),
            mul.as_mut_ptr(),
            div.as_mut_ptr(),
            n,
            block_size,
            grid_size,
        );
    }

    let add_h = add.to_host();
    let sub_h = sub.to_host();
    let mul_h = mul.to_host();
    let div_h = div.to_host();

    for (i, (a, b, add, sub, mul, div)) in izip!(a_h, b_h, add_h, sub_h, mul_h, div_h).enumerate() {
        assert_eq!(a + b, add, "i={}", i);
        assert_eq!(a - b, sub, "i={}", i);
        assert_eq!(a * b, mul, "i={}", i);
        assert_eq!(a / b, div, "i={}", i);
    }
}
