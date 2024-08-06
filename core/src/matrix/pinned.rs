use p3_matrix::dense::RowMajorMatrix;

use crate::{
    cuda_runtime::{stream::CudaStream, PinBuffer},
    device::{error::CudaError, memory::ToDevice},
};

use super::RowMajorMatrixDevice;

pub struct PinRowMajorMatrix<T> {
    values: PinBuffer<T>,
    width: usize,
}

impl<T> PinRowMajorMatrix<T> {
    pub fn new(values: PinBuffer<T>, width: usize) -> Self {
        debug_assert!(values.len() % width == 0);
        Self { values, width }
    }

    pub fn register(matrix: RowMajorMatrix<T>) -> Result<Self, CudaError> {
        let values = PinBuffer::register(matrix.values)?;
        Ok(Self::new(values, matrix.width))
    }

    pub fn height(&self) -> usize {
        if self.width == 0 {
            return 0;
        }
        self.values.len() / self.width
    }
}

impl<T: Copy + Send + Sync> ToDevice for PinRowMajorMatrix<T> {
    type DeviceType = RowMajorMatrixDevice<T>;

    fn to_device_async(&self, stream: &CudaStream) -> Result<Self::DeviceType, CudaError> {
        let values = self.values.to_device_async(stream)?;
        Ok(RowMajorMatrixDevice::new(values, self.width))
    }
}
