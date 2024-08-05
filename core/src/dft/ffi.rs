use p3_baby_bear::BabyBear;

use crate::device::error::CudaRustError;

extern "C" {
    pub(crate) fn sppark_init() -> CudaRustError;

    pub(crate) fn batch_lde_shift(
        d_inout: *mut BabyBear,
        lg_domain_size: u32,
        lg_blowup: u32,
        shift: BabyBear,
        poly_count: u32,
        is_bit_rev: bool,
    ) -> CudaRustError;

}
