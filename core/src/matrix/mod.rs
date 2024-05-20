use p3_matrix::dense::RowMajorMatrix;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::device::buffer::DeviceBuffer;
use crate::device::buffer::ToDevice;

/// A matrix stored on the device in row major form.
#[derive(Debug)]
#[repr(C)]
pub struct RowMajorMatrixDevice<T: Copy> {
    pub values: DeviceBuffer<T>,
    pub width: usize,
}

/// A view of a matrix stored on the device in row major form.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RowMajorMatrixViewDevice<T> {
    pub values: *const T,
    pub width: usize,
    pub height: usize,
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

    pub fn view(&self) -> RowMajorMatrixViewDevice<T> {
        RowMajorMatrixViewDevice {
            values: self.values.as_ptr(),
            width: self.width,
            height: self.values.len() / self.width,
        }
    }

    pub fn to_host(&self) -> RowMajorMatrix<T> {
        RowMajorMatrix::new(self.values.to_host(), self.width)
    }

    pub fn height(&self) -> usize {
        self.values.len() / self.width
    }
}
