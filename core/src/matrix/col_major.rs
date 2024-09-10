use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::cuda_runtime::stream::CudaStream;
use crate::device::error::CudaError;
use crate::device::memory::{ToDevice, ToHost};
use crate::device::DeviceBuffer;

use super::ffi::{self, transpose_naive};
use super::{DeviceMatrix, MatrixViewDevice, MatrixViewMutDevice};

/// A matrix stored on the device in column major form.
#[derive(Debug)]
#[repr(C)]
pub struct ColMajorMatrixDevice<T: Copy> {
    pub values: DeviceBuffer<T>,
    pub height: usize,
}

impl<T: Default + Copy + Send + Sync> ColMajorMatrixDevice<T> {
    pub fn new(values: DeviceBuffer<T>, height: usize) -> Self {
        Self { values, height }
    }

    pub fn null() -> Self {
        Self { values: DeviceBuffer::with_capacity(0).unwrap(), height: 1 }
    }

    pub const fn stream(&self) -> &CudaStream {
        self.values.stream()
    }

    pub fn with_capacity(width: usize, height: usize) -> Result<Self, CudaError> {
        let buffer = DeviceBuffer::with_capacity(width * height)?;
        Ok(Self::new(buffer, height))
    }

    pub fn with_capacity_in(
        width: usize,
        height: usize,
        stream: &CudaStream,
    ) -> Result<Self, CudaError> {
        let buffer = DeviceBuffer::with_capacity_in(width * height, stream)?;
        Ok(Self::new(buffer, height))
    }

    pub fn to_host_naive(&self) -> RowMajorMatrix<T> {
        RowMajorMatrix::new(self.values.to_host(), self.height).transpose()
    }

    /// # Safety
    ///
    /// TODO
    pub unsafe fn set_width(&mut self, width: usize) {
        self.values.set_len(width * self.height);
    }

    /// # Safety
    ///
    /// See [Self::set_height]
    pub unsafe fn set_max_width(&mut self) {
        self.values.set_max_len();
    }

    pub fn dummy(width: usize, height: usize) -> (RowMajorMatrix<T>, Self)
    where
        Standard: Distribution<T>,
    {
        let mut rng = rand::thread_rng();
        let data = (0..width * height).map(|_| rng.gen()).collect::<Vec<_>>();
        let device = ColMajorMatrixDevice::new(data.to_device().unwrap(), height);
        let host = RowMajorMatrix::new(data, height).transpose();
        (host, device)
    }

    /// Returns a view of the matrix in column major form.
    pub fn view(&self) -> MatrixViewDevice<T> {
        MatrixViewDevice {
            values: self.values.as_ptr(),
            width: self.width(),
            height: self.height(),
            row_major: false,
        }
    }

    /// Returns a mutable view of the matrix in column major form.
    pub fn view_mut(&mut self) -> MatrixViewMutDevice<T> {
        MatrixViewMutDevice {
            values: self.values.as_mut_ptr(),
            width: self.width(),
            height: self.height(),
            row_major: false,
        }
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.values.len() / self.height
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// # Safety
    ///
    /// The memory returened by this function is only partially initialized.
    pub unsafe fn embed_as_blowup(
        &self,
        log_blowup: usize,
    ) -> Result<ColMajorMatrixDevice<T>, CudaError> {
        let mut blowup_values =
            DeviceBuffer::with_capacity_in(self.values.len() << log_blowup, self.stream())?;
        unsafe { blowup_values.set_max_len() };

        let blowup_height = self.height << log_blowup;

        // Copy the columns from the source buffer into the correct place in the destination buffer.
        for j in 0..self.width() {
            let src = &self.values[j * self.height..(j + 1) * self.height];
            let dst = &mut blowup_values
                [j * blowup_height + blowup_height - self.height..(j + 1) * blowup_height];
            dst.copy_from_device(src, self.stream())?;
        }

        Ok(ColMajorMatrixDevice::new(blowup_values, blowup_height))
    }
}

impl ColMajorMatrixDevice<BabyBear> {
    pub fn bit_reverse_rows(&mut self) -> Result<(), CudaError> {
        assert_eq!(self.height, 1 << self.height.ilog2(), "height must be a power of 2");
        unsafe {
            ffi::reverse_bits_batch(
                self.values.as_mut_ptr(),
                self.values.as_ptr(),
                self.height.ilog2(),
                self.width(),
                self.stream().handle(),
            )
        }
        .to_result()
    }

    pub fn vertically_strided(
        &self,
        stride: usize,
        offset: usize,
    ) -> Result<ColMajorMatrixDevice<BabyBear>, CudaError> {
        assert_eq!(self.height % stride, 0, "height must be a multiple of stride");
        let mut strided_values =
            DeviceBuffer::with_capacity_in(self.values.len() / stride, self.stream()).unwrap();
        unsafe { strided_values.set_max_len() };

        let mut output = ColMajorMatrixDevice::new(strided_values, self.height / stride);
        unsafe {
            ffi::strided_matrix(
                output.view_mut(),
                self.view(),
                stride,
                offset,
                self.stream().handle(),
            )
        };

        Ok(output)
    }
}

impl ToHost for ColMajorMatrixDevice<BabyBear> {
    type HostType = RowMajorMatrix<BabyBear>;

    /// Returns a host copy of the matrix in row major form.
    fn to_host(&self) -> Self::HostType {
        let mut ret_values =
            DeviceBuffer::with_capacity_in(self.height() * self.width(), self.stream()).unwrap();
        unsafe {
            ret_values.set_max_len();
            transpose_naive(ret_values.as_mut_ptr(), self.view(), self.stream().handle())
        };
        RowMajorMatrix::new(ret_values.to_host(), self.width())
    }
}

impl<T: Default + Copy + Send + Sync> DeviceMatrix<T> for ColMajorMatrixDevice<T> {
    fn width(&self) -> usize {
        self.width()
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

    fn stream(&self) -> &CudaStream {
        self.stream()
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_matrix::bitrev::BitReversableMatrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use rand::thread_rng;

    use crate::{cuda_runtime::sync_device, device::memory::ToHost};

    use crate::device::memory::ToDevice;

    use super::*;

    #[test]
    fn test_col_major_to_host() {
        let height = 1 << 18;
        let width = 200;

        let mut rng = thread_rng();
        let values = (0..width * height)
            .map(|_| rng.gen::<BabyBear>())
            .collect::<Vec<_>>()
            .to_device()
            .unwrap();

        let matrix = ColMajorMatrixDevice::new(values, height);

        sync_device().unwrap();

        let time = std::time::Instant::now();
        let matrix_host_naive = matrix.to_host_naive();
        println!("Naive time: {:?}", time.elapsed());

        let time = std::time::Instant::now();
        let matrix_host = matrix.to_host();
        println!("time: {:?}", time.elapsed());

        for (val, exp) in matrix_host.values.into_iter().zip(matrix_host_naive.values) {
            assert_eq!(val, exp);
        }
    }

    #[test]
    fn test_bit_reverse_rows() {
        let height = 1 << 16;
        let width = 100;

        let mut rng = thread_rng();
        let host_matrix = RowMajorMatrix::<BabyBear>::rand(&mut rng, height, width);

        let mut device_matrix = host_matrix.to_device().unwrap().to_column_major();
        device_matrix.bit_reverse_rows().unwrap();

        let host_matrix_reversed = host_matrix.bit_reverse_rows().to_row_major_matrix();

        let device_matrix_back = device_matrix.to_host();

        for (val, exp) in host_matrix_reversed.values.into_iter().zip(device_matrix_back.values) {
            assert_eq!(val, exp);
        }
    }

    #[test]
    fn test_strided() {
        let height = 1 << 16;
        let width = 100;
        let stride = 1 << 4;

        let mut rng = thread_rng();
        let host_matrix = RowMajorMatrix::<BabyBear>::rand(&mut rng, height, width);

        let device_matrix = host_matrix.to_device().unwrap().to_column_major();

        for offset in 0..stride {
            let strided_d = device_matrix.vertically_strided(stride, offset).unwrap();
            let mat_h = host_matrix.clone();
            let host_matrix_strided =
                mat_h.vertically_strided(stride, offset).to_row_major_matrix();

            let device_matrix_back = strided_d.to_host();

            for (val, exp) in host_matrix_strided.values.into_iter().zip(device_matrix_back.values)
            {
                assert_eq!(val, exp);
            }
        }
    }
}
