/// A `Task` is used to advertise work for the device.
pub(super) trait Task {
    /// Unsafe: this may be called from a different thread than the one
    /// which scheduled the job, so the implementer must ensure the
    /// appropriate traits are met, whether `Send`, `Sync`, or both.
    unsafe fn execute(this: *const ());
}

pub struct TaskRef {
    pointer: *const (),
    execute_fn: unsafe fn(*const ()),
}

unsafe impl Send for TaskRef {}
unsafe impl Sync for TaskRef {}

impl TaskRef {
    /// Unsafe: caller asserts that `data` will remain valid until the
    /// job is executed.
    pub(super) unsafe fn new<T>(data: *const T) -> TaskRef
    where
        T: Task,
    {
        // erase types:
        TaskRef { pointer: data as *const (), execute_fn: <T as Task>::execute }
    }

    // /// Returns an opaque handle that can be saved and compared,
    // /// without making `JobRef` itself `Copy + Eq`.
    // #[inline]
    // pub(super) fn id(&self) -> impl Eq {
    //     (self.pointer, self.execute_fn)
    // }

    #[inline]
    pub(super) unsafe fn execute(self) {
        (self.execute_fn)(self.pointer)
    }
}

pub struct HeapTask<F>
where
    F: FnOnce() + Send,
{
    task: F,
}

impl<F> HeapTask<F>
where
    F: FnOnce() + Send,
{
    pub(super) fn new(task: F) -> Box<Self> {
        Box::new(Self { task })
    }

    pub(super) unsafe fn into_task_ref(self) -> TaskRef {
        TaskRef::new(Box::into_raw(Box::new(self)))
    }
}

impl<F: FnOnce() + Send> Task for HeapTask<F> {
    unsafe fn execute(this: *const ()) {
        let this = Box::from_raw(this as *mut Self);
        (this.task)();
    }
}
