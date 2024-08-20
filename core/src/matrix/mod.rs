mod col_major;
mod pinned;
mod row_major;
mod view;

pub use col_major::*;
pub use pinned::*;
pub use row_major::*;
pub use view::*;

pub trait DeviceMatrix<T: Copy> {
    /// The number of columns in the matrix.
    fn width(&self) -> usize;

    /// The number of rows in the matrix.
    fn height(&self) -> usize;

    /// A view of the matrix.
    fn view(&self) -> MatrixViewDevice<T>;

    /// A mutable view of the matrix.
    fn view_mut(&mut self) -> MatrixViewMutDevice<T>;
}

pub(super) mod ffi {
    use p3_baby_bear::BabyBear;

    use crate::{cuda_runtime::stream::CudaStreamHandle, device::error::CudaRustError};

    use super::{MatrixViewDevice, MatrixViewMutDevice};

    #[link_name = "matrix_transpose"]
    #[allow(unused_attributes)]
    extern "C" {
        pub fn transpose_naive(
            output: *mut BabyBear,
            input: MatrixViewDevice<BabyBear>,
            stream: CudaStreamHandle,
        );

        pub fn transpose_blowup_naive(
            output: *mut BabyBear,
            input: MatrixViewDevice<BabyBear>,
            log_blowup: usize,
            stream: CudaStreamHandle,
        );
    }

    #[link_name = "matrix_strided"]
    #[allow(unused_attributes)]
    extern "C" {
        pub fn strided_matrix(
            output: MatrixViewMutDevice<BabyBear>,
            input: MatrixViewDevice<BabyBear>,
            stride: usize,
            offset: usize,
            stream: CudaStreamHandle,
        );

        #[allow(dead_code)]
        pub fn split_rows(
            outputs: *mut MatrixViewMutDevice<BabyBear>,
            input: MatrixViewDevice<BabyBear>,
            stride: usize,
            stream: CudaStreamHandle,
        );
    }

    extern "C" {
        pub fn reverse_bits_batch(
            output: *mut BabyBear,
            input: *const BabyBear,
            lg_domain_size: u32,
            poly_count: usize,
        ) -> CudaRustError;
    }
}
