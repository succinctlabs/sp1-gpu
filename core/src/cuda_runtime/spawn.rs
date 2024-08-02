use std::fmt;

use super::{
    current_device,
    task::{HeapTask, TaskRef},
};

pub struct JoinHandle<T> {
    rx: oneshot::Receiver<T>,
}

pub(super) fn spawn_ref(task: TaskRef) {
    let device_sender = current_device();
    device_sender.send(task).unwrap();
}

pub fn spawn<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    unsafe { spawn_unchecked(f) }
}

pub(crate) unsafe fn spawn_unchecked<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T + Send,
    T: Send,
{
    let (tx, rx) = oneshot::channel::<T>();

    let task = HeapTask::new(move || {
        let result = f();
        tx.send(result).unwrap();
    });
    spawn_ref(task.into_task_ref());

    JoinHandle { rx }
}

impl<T> JoinHandle<T> {
    pub fn sync_join(self) -> Result<T, JoinError> {
        self.rx.recv().map_err(|_| JoinError)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct JoinError;

impl fmt::Display for JoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "Failed to join task".fmt(f)
    }
}

impl std::error::Error for JoinError {}
