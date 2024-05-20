use p3_matrix::dense::RowMajorMatrix;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::device::buffer::DeviceBuffer;
use crate::device::buffer::ToDevice;

use super::{MatrixViewDevice, MatrixViewMutDevice};

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
