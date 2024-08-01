mod pinned;
mod stream;
mod sync;

pub use pinned::*;
pub use stream::*;
pub use sync::*;

pub trait RawPointer {
    type Data;

    fn as_ptr(&self) -> *const Self::Data;
    fn as_mut_ptr(&mut self) -> *mut Self::Data;

    fn free(&mut self);
}
