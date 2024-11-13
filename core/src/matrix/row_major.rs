use moongate_bloc::alloc::Allocator;
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;

use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

use crate::{
    cuda_runtime::{stream::CudaStream, CudaSync, DeviceAllocator},
    device::{
        error::CudaError,
        memory::{ToDeviceIn, ToDevice, ToHost},
        DeviceBuffer,
    },
};

use super::{
    ffi::{transpose_blowup_naive, transpose_naive},
    ColMajorMatrixDevice, DeviceMatrix, MatrixViewDevice, MatrixViewMutDevice,
};

/// A matrix stored on the device in row major form.
#[derive(Debug)]
#[repr(C)]
pub struct RowMajorMatrixDevice<T: Copy, A: Allocator = CudaStream> {
    pub values: DeviceBuffer<T, A>,
    pub width: usize,
}

impl<T: Copy> RowMajorMatrixDevice<T> {
    pub fn dummy(width: usize, height: usize) -> (RowMajorMatrix<T>, Self)
    where
        Standard: Distribution<T>,
        T: Send + Sync,
    {
        let mut rng = rand::thread_rng();
        let data = (0..width * height).map(|_| rng.gen()).collect::<Vec<_>>();
        let device = RowMajorMatrixDevice::new(data.to_device().unwrap(), width);
        let host = RowMajorMatrix::new(data, width);
        (host, device)
    }
}

impl<T: Copy, A: Allocator + CudaSync> CudaSync for RowMajorMatrixDevice<T, A> {
    fn stream(&self) -> &CudaStream {
        self.values.stream()
    }
}

impl<T: Copy, A: Allocator> RowMajorMatrixDevice<T, A> {
    pub fn new(values: DeviceBuffer<T, A>, width: usize) -> Self {
        Self { values, width }
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

    pub fn to_host(&self) -> RowMajorMatrix<T>
    where
        T: Send + Sync,
        A: CudaSync,
    {
        RowMajorMatrix::new(self.values.to_host(), self.width)
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.values.len() / self.width
    }

    #[inline]
    pub const fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn allocator(&self) -> &A {
        self.values.allocator()
    }
}

impl<T: Copy + Send + Sync, A: DeviceAllocator> DeviceMatrix<T> for RowMajorMatrixDevice<T, A> {
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

impl<A: DeviceAllocator> RowMajorMatrixDevice<BabyBear, A> {
    pub fn to_column_major(&self) -> ColMajorMatrixDevice<BabyBear, A> {
        let mut ret_values = DeviceBuffer::with_capacity_in(
            self.height() * self.width(),
            self.values.allocator().clone(),
        )
        .unwrap();
        unsafe { transpose_naive(ret_values.as_mut_ptr(), self.view(), self.stream().handle()) };
        unsafe { ret_values.set_max_len() };

        ColMajorMatrixDevice::new(ret_values, self.height())
    }

    pub fn to_column_major_blowup_in<B: Allocator>(
        &self,
        log_blowup: usize,
        alloc: B,
    ) -> ColMajorMatrixDevice<BabyBear, B> {
        let mut ret_values =
            DeviceBuffer::with_capacity_in(self.values.len() << log_blowup, alloc).unwrap();
        unsafe {
            transpose_blowup_naive(
                ret_values.as_mut_ptr(),
                self.view(),
                log_blowup,
                self.values.stream().handle(),
            )
        };
        unsafe { ret_values.set_max_len() };

        ColMajorMatrixDevice::new(ret_values, self.height() << log_blowup)
    }

    pub fn to_column_major_blowup(&self, log_blowup: usize) -> ColMajorMatrixDevice<BabyBear, A> {
        self.to_column_major_blowup_in(log_blowup, self.allocator().clone())
    }
}

impl<T: Copy + Send + Sync> ToHost for RowMajorMatrixDevice<T> {
    type HostType = RowMajorMatrix<T>;

    fn to_host(&self) -> Self::HostType {
        RowMajorMatrix::new(self.values.to_host(), self.width)
    }
}

impl<T: Copy + Send + Sync, A: Allocator + CudaSync> ToDeviceIn<A> for RowMajorMatrix<T> {
    type DeviceType = RowMajorMatrixDevice<T, A>;

    fn to_device_in(&self, alloc: A) -> Result<Self::DeviceType, CudaError> {
        let values = self.values.to_device_in(alloc)?;
        Ok(RowMajorMatrixDevice::new(values, self.width))
    }
}

#[cfg(test)]
mod tests {

    use p3_matrix::Matrix;

    use crate::device::memory::ToHost;

    use super::*;

    #[test]
    fn test_row_major_to_column_major() {
        let height = 1 << 21;
        let width = 600;

        let (mat_h, mat_d) = RowMajorMatrixDevice::<BabyBear>::dummy(width, height);

        let start = crate::time::CudaInstant::now().unwrap();
        let mat_d_col = mat_d.to_column_major();
        let cpu_time = start.elapsed().unwrap();
        println!("time: {:?}", cpu_time);

        let mat_d_h = mat_d_col.to_host();

        for (val, exp) in mat_d_h.values.into_iter().zip(mat_h.values) {
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

        assert_eq!(mad_d_col.height(), ext_height);
        assert_eq!(mad_d_col.width(), width);

        // Check the transposed matrix.
        let mad_d_to_h = mad_d_col.values.to_host();

        for i in 0..height {
            for j in 0..width {
                assert_eq!(
                    mad_d_to_h[j * ext_height + ext_height - height + i],
                    matrix_h.values[i * width + j]
                );
            }
        }

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
