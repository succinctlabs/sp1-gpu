use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;

use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::device::buffer::DeviceBuffer;
use crate::device::buffer::ToDevice;

use super::ffi::{transpose_blowup_naive, transpose_naive};
use super::{ColMajorMatrixDevice, DeviceMatrix, MatrixViewDevice, MatrixViewMutDevice};

/// A matrix stored on the device in row major form.
#[derive(Debug)]
#[repr(C)]
pub struct RowMajorMatrixDevice<T: Copy> {
    pub values: DeviceBuffer<T>,
    pub width: usize,
}

impl<T: Copy + Send + Sync> RowMajorMatrixDevice<T> {
    pub fn new(values: DeviceBuffer<T>, width: usize) -> Self {
        Self { values, width }
    }

    pub fn dummy(width: usize, height: usize) -> (RowMajorMatrix<T>, Self)
    where
        Standard: Distribution<T>,
    {
        let mut rng = rand::thread_rng();
        let data = (0..width * height).map(|_| rng.gen()).collect::<Vec<_>>();
        let device = RowMajorMatrixDevice::new(data.to_device(), width);
        let host = RowMajorMatrix::new(data, width);
        (host, device)
    }

    pub fn view(&self) -> MatrixViewDevice<T> {
        MatrixViewDevice {
            values: self.values.as_ptr(),
            width: self.width,
            height: self.values.len() / self.width,
            row_major: true,
        }
    }

    pub fn view_mut(&mut self) -> MatrixViewMutDevice<T> {
        MatrixViewMutDevice {
            values: self.values.as_mut_ptr(),
            width: self.width,
            height: self.values.len() / self.width,
            row_major: true,
        }
    }

    pub fn to_host(&self) -> RowMajorMatrix<T> {
        RowMajorMatrix::new(self.values.to_host(), self.width)
    }

    pub fn height(&self) -> usize {
        self.values.len() / self.width
    }

    pub fn width(&self) -> usize {
        self.width
    }
}

impl<T: Copy + Send + Sync> DeviceMatrix<T> for RowMajorMatrixDevice<T> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height()
    }

    fn view(&self) -> MatrixViewDevice<T> {
        self.view()
    }

    fn view_mut(&mut self) -> MatrixViewMutDevice<T> {
        self.view_mut()
    }
}

impl RowMajorMatrixDevice<BabyBear> {
    pub fn to_column_major(&self) -> ColMajorMatrixDevice<BabyBear> {
        let mut ret_values = DeviceBuffer::with_capacity(self.height() * self.width());
        unsafe { transpose_naive(ret_values.as_mut_ptr(), self.view()) };
        unsafe { ret_values.set_max_len() };

        ColMajorMatrixDevice::new(ret_values, self.height())
    }

    pub fn to_column_major_blowup(&self, log_blowup: usize) -> ColMajorMatrixDevice<BabyBear> {
        let mut ret_values = DeviceBuffer::with_capacity(self.values.len() << log_blowup);
        unsafe { transpose_blowup_naive(ret_values.as_mut_ptr(), self.view(), log_blowup) };
        unsafe { ret_values.set_max_len() };

        ColMajorMatrixDevice::new(ret_values, self.height())
    }
}

#[cfg(test)]
mod tests {

    use p3_matrix::Matrix;

    use super::*;

    #[test]
    fn test_row_major_to_column_major() {
        let height = 1 << 22;
        let width = 600;

        let (mat_h, mat_d) = RowMajorMatrixDevice::<BabyBear>::dummy(width, height);

        let start = crate::time::CudaInstant::now().unwrap();
        let mat_d_col = mat_d.to_column_major();
        let cpu_time = start.elapsed().unwrap();
        println!("time: {:?}", cpu_time);

        let mat_d_values = mat_d_col.values.to_host();
        let mat_h_transposed = mat_h.transpose();

        for (val, exp) in mat_d_values.into_iter().zip(mat_h_transposed.values) {
            assert_eq!(val, exp);
        }
    }

    #[test]
    fn test_blowup_transpose() {
        let height = 1 << 16;
        let width = 600;
        let log_blowup = 1;

        let ext_height = height << log_blowup;

        let (matrix_h, matrix_d) = RowMajorMatrixDevice::<BabyBear>::dummy(width, height);

        // Transpose the matrix.
        let start = crate::time::CudaInstant::now().unwrap();
        let mad_d_col = matrix_d.to_column_major_blowup(log_blowup);
        let time = start.elapsed().unwrap();
        println!("time: {:?}", time);

        // Check the transposed matrix.
        let mad_d_to_h = mad_d_col.values.to_host();

        for (j, col) in mad_d_to_h.chunks(ext_height).enumerate() {
            for i in 0..height {
                assert_eq!(
                    col[ext_height - height + i],
                    matrix_h.get(i, j),
                    "row {}, col {}",
                    i,
                    j
                );
            }
        }
    }
}
