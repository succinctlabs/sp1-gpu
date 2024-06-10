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

impl<T> MatrixViewDevice<T> {
    pub fn null(row_major: bool) -> Self {
        Self {
            values: std::ptr::null(),
            width: 0,
            height: 0,
            row_major,
        }
    }
}
