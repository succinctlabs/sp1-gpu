/// A view of a matrix stored on the device in row major form.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MatrixViewDevice<'a, T> {
    pub values: *const T,
    pub width: usize,
    pub height: usize,
    pub row_major: bool,
    pub(crate) _marker: std::marker::PhantomData<&'a [T]>,
}

/// A view of a matrix stored on the device in row major form.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct MatrixViewMutDevice<'a, T> {
    pub values: *mut T,
    pub width: usize,
    pub height: usize,
    pub row_major: bool,
    pub(crate) _marker: std::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T> MatrixViewDevice<'a, T> {
    pub fn null(row_major: bool) -> Self {
        Self {
            values: std::ptr::null(),
            width: 0,
            height: 0,
            row_major,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn values_ptr(&self) -> *const T {
        self.values
    }
}

impl<'a, T> MatrixViewMutDevice<'a, T> {
    pub fn values_ptr(&self) -> *mut T {
        self.values
    }

    pub fn values_mut_ptr(&mut self) -> *mut T {
        self.values
    }
}

// TODO: remove the need for this in the future.
impl<T: Copy> Copy for MatrixViewMutDevice<'_, T> {}
