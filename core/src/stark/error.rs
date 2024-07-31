use crate::device::error::CudaError;

pub enum StarkProverError {
    CudaError(CudaError),
}
