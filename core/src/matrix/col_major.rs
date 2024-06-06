use p3_matrix::dense::RowMajorMatrix;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::device::buffer::DeviceBuffer;
use crate::device::error::CudaError;
use crate::device::memory::{ToDevice, ToHost};

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

    pub fn with_capacity(width: usize, height: usize) -> Self {
        let buffer = DeviceBuffer::with_capacity(width * height);
        Self::new(buffer, height)
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
        let device = ColMajorMatrixDevice::new(data.to_device(), height);
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

    /// Returns a host copy of the matrix in row major form.
    pub fn to_host(&self) -> RowMajorMatrix<T> {
        RowMajorMatrix::new(self.values.to_host(), self.height).transpose()
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
        let mut blowup_values = DeviceBuffer::with_capacity(self.values.len() << log_blowup);
        unsafe { blowup_values.set_max_len() };

        let blowup_height = self.height << log_blowup;

        // Copy the columns from the source buffer into the correct place in the destination buffer.
        for j in 0..self.width() {
            let src = &self.values[j * self.height..(j + 1) * self.height];
            let dst = &mut blowup_values
                [j * blowup_height + blowup_height - self.height..(j + 1) * blowup_height];
            dst.copy_from_device(src)?;
        }

        Ok(ColMajorMatrixDevice::new(blowup_values, blowup_height))
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
}
