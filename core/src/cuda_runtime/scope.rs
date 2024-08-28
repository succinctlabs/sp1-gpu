use std::{
    marker::PhantomData,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use super::{spawn_unchecked, JoinHandle};

pub struct Scope<'scope, 'env: 'scope> {
    task_count: Arc<AtomicUsize>,
    scope: PhantomData<&'scope mut &'scope ()>,
    env: PhantomData<&'env mut &'env ()>,
}

pub fn scope<'env, F, T>(f: F) -> T
where
    F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> T,
{
    let scope =
        Scope { task_count: Arc::new(AtomicUsize::new(0)), scope: PhantomData, env: PhantomData };

    let result = f(&scope);

    // Wait for all tasks to complete
    while scope.task_count.load(Ordering::Relaxed) > 0 {
        std::thread::yield_now(); // Yield to allow other threads to run
    }

    result
}

impl<'scope, 'env: 'scope> Scope<'scope, 'env> {
    pub fn spawn<F, T>(&self, f: F) -> JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
        let task_count = self.task_count.clone();
        task_count.fetch_add(1, Ordering::Relaxed);

        unsafe {
            spawn_unchecked(move || {
                let result = f();
                task_count.fetch_sub(1, Ordering::Relaxed);
                result
            })
        }
    }
}
