use p3_baby_bear::BabyBear;

extern "C" {
    pub(crate) fn coset_lde(d_data: *mut BabyBear, log_degree: usize, log_blowup: usize);

    pub(crate) fn dft(d_data: *mut BabyBear, log_degree: usize);
}
