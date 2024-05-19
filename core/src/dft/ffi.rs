use p3_baby_bear::BabyBear;

use crate::device::error::CudaRustError;

extern "C" {
    pub(crate) fn sppark_init() -> CudaRustError;

    pub(crate) fn sppark_batch_expand(
        d_out: *mut BabyBear,
        d_in: *const BabyBear,
        lg_domain_size: u32,
        lg_blowup: u32,
        poly_count: u32,
    ) -> CudaRustError;

    pub(crate) fn batch_NTT(
        d_inout: *mut BabyBear,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> CudaRustError;

    pub(crate) fn batch_iNTT(
        d_inout: *mut BabyBear,
        lg_domain_size: u32,
        poly_count: u32,
    ) -> CudaRustError;
}
