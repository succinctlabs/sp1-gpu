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
    pub const fn new(values: PinBuffer<T>, width: usize) -> Self {
        Self { values, width }
    }

    pub fn register(matrix: RowMajorMatrix<T>) -> Result<Self, CudaError> {
        let values = PinBuffer::register(matrix.values)?;
        Ok(Self::new(values, matrix.width))
    }
}

impl<T: Copy + Send + Sync> ToDevice for PinRowMajorMatrix<T> {
    type DeviceType = RowMajorMatrixDevice<T>;

    fn to_device_async(&self, stream: &CudaStream) -> Result<Self::DeviceType, CudaError> {
        let values = self.values.to_device_async(stream)?;
        Ok(RowMajorMatrixDevice::new(values, self.width))
    }
}
