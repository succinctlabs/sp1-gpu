pub mod buffer;
pub mod error;
pub(crate) mod ffi;
pub mod memory;
pub mod slice;

#[repr(C)]
pub struct Device {
    cuda_id: usize,
}

impl Device {}
