mod pinned;
mod stream;
mod sync;

pub use pinned::*;
pub use stream::*;
pub use sync::*;

pub trait RawPointer<T> {
    fn as_ptr(&self) -> *const T;
    fn as_mut_ptr(&mut self) -> *mut T;

    fn free(&mut self);
}
