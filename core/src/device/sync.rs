use std::{
    borrow::Borrow,
    ops::{Deref, DerefMut},
};

use crate::{
    device::error::CudaError,
    matrix::{DeviceMatrix, MatrixViewDevice},
};
use crate::{matrix::MatrixViewMutDevice, runtime::sync_default_stream};

#[repr(transparent)]
pub struct CudaSync<T>(T);

unsafe impl<T> Send for CudaSync<T> {}
unsafe impl<T> Sync for CudaSync<T> {}

impl<T> CudaSync<T> {
    pub fn new(value: T) -> Result<Self, CudaError> {
        sync_default_stream()?;
        Ok(Self(value))
    }

    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> Borrow<T> for CudaSync<T> {
    fn borrow(&self) -> &T {
        &self.0
    }
}

impl<T> AsRef<T> for CudaSync<T> {
    fn as_ref(&self) -> &T {
        &self.0
    }
}

impl<T> Deref for CudaSync<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Copy, M: DeviceMatrix<T>> DeviceMatrix<T> for CudaSync<M> {
    fn width(&self) -> usize {
        self.0.width()
    }

    fn height(&self) -> usize {
        self.0.height()
    }

    fn view(&self) -> MatrixViewDevice<T> {
        self.0.view()
    }

    fn view_mut(&mut self) -> MatrixViewMutDevice<T> {
        panic!("Unsafe")
    }
}
