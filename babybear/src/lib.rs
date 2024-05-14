use p3_baby_bear::BabyBear;

extern "C" {
    fn add_baby_bear(a: *const BabyBear, b: *const BabyBear, c: *mut BabyBear, n: usize);
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use rand::{thread_rng, Rng};

    use crate::add_baby_bear;

    #[test]
    fn test_baby_bear_add() {
        let n = 10000;

        let mut rng = thread_rng();

        let a = (0..n).map(|_| rng.gen::<BabyBear>()).collect::<Vec<_>>();
        let b = (0..n).map(|_| rng.gen::<BabyBear>()).collect::<Vec<_>>();
        let mut c = vec![BabyBear::zero(); n];

        unsafe {
            add_baby_bear(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n);
        }

        for ((ai, bi), ci) in a.iter().zip(b.iter()).zip(c.iter()) {
            assert_eq!(*ci, *ai + *bi);
        }
    }
}
