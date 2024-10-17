use p3_baby_bear::BabyBear;

use crate::{cuda_runtime::stream::CudaStreamHandle, device::error::CudaRustError};

extern "C" {
    pub(crate) fn sppark_init(stream: CudaStreamHandle) -> CudaRustError;

    pub(crate) fn batch_lde_shift(
        d_inout: *mut BabyBear,
        lg_domain_size: u32,
        lg_blowup: u32,
        shift: BabyBear,
        poly_count: u32,
        is_bit_rev: bool,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    pub(crate) fn batch_NTT(
        d_inout: *mut BabyBear,
        lg_domain_size: u32,
        poly_count: u32,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    pub(crate) fn batch_iNTT(
        d_inout: *mut BabyBear,
        lg_domain_size: u32,
        poly_count: u32,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}
