use crate::cuda_runtime::stream::CudaStream;

pub struct DeviceStreamPointer<T> {
    ptr: *mut T,
    stream: CudaStream,
}
