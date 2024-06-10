mod col_major;
mod row_major;
mod view;

pub use col_major::*;
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

    use crate::device::error::CudaRustError;

    use super::MatrixViewDevice;

    extern "C" {
        pub fn transpose_naive(output: *mut BabyBear, input: MatrixViewDevice<BabyBear>);

        pub fn transpose_blowup_naive(
            output: *mut BabyBear,
            input: MatrixViewDevice<BabyBear>,
            log_blowup: usize,
        );

        pub fn reverse_bits_batch(
            output: *mut BabyBear,
            input: *const BabyBear,
            lg_domain_size: u32,
            poly_count: usize,
        ) -> CudaRustError;
    }
}
