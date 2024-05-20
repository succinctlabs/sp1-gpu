/// A view of a matrix stored on the device in row major form.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MatrixViewDevice<T> {
    pub values: *const T,
    pub width: usize,
    pub height: usize,
    pub row_major: bool,
}

/// A view of a matrix stored on the device in row major form.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MatrixViewMutDevice<T> {
    pub values: *mut T,
    pub width: usize,
    pub height: usize,
    pub row_major: bool,
}
