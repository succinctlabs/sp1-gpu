use std::{alloc::Layout, mem::ManuallyDrop, ptr::NonNull, rc::Rc, sync::Mutex};

use crate::{
    alloc::{AllocError, Allocator, DeviceMemory},
    bump::Bump,
};

#[derive(Default, Debug)]
pub struct Herd<A: Allocator>(Mutex<Vec<Box<Bump<A>>>>);

impl<A: Allocator> Herd<A> {
    pub fn new() -> Self {
        Herd(Mutex::new(vec![]))
    }

    /// Borrows a member allocator from this herd.
    ///
    /// As the [`Herd`] is [`Sync`], it is possible to call this from the worker threads. The
    /// [`Member`] is a proxy around [`Bump`], allowing to allocate objects with lifetime of the
    /// [`Herd`] (therefore, the allocated objects can live longer than the [`Member`] itself).
    ///
    /// # Performance note
    ///
    /// This is not cheap and is not expected to happen often. It contains a mutex.
    ///
    /// The expected usage pattern is that each worker thread (or similar entity) grabs one
    /// allocator *once*, at its start and uses it through its lifetime, not that it would call
    /// `get` on each allocation.
    pub fn get(&self) -> Member<'_, A> {
        let mut lock = self.0.lock().unwrap();
        let bump = lock.pop().expect("herd is empty");
        let inner = MemberInner { arena: ManuallyDrop::new(bump), owner: self };
        Member(Rc::new(inner))
    }
}

#[derive(Debug)]
struct MemberInner<'h, A: Allocator> {
    arena: ManuallyDrop<Box<Bump<A>>>,
    owner: &'h Herd<A>,
}

#[derive(Debug, Clone)]
pub struct Member<'h, A: Allocator>(Rc<MemberInner<'h, A>>);

impl<'h, A> DeviceMemory for Member<'h, A>
where
    A: Allocator,
{
    #[inline]
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
    ) -> Result<(), AllocError> {
        self.0.arena.copy_nonoverlapping(src, dst, size)
    }

    #[inline]
    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), AllocError> {
        self.0.arena.write_bytes(dst, value, size)
    }
}

unsafe impl<'h, A> Allocator for Member<'h, A>
where
    A: Allocator,
{
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.arena.allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.arena.allocate_zeroed(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { self.0.arena.deallocate(ptr, layout) }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { self.0.arena.grow(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { self.0.arena.grow_zeroed(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { self.0.arena.shrink(ptr, old_layout, new_layout) }
    }
}

impl<A: Allocator> Drop for MemberInner<'_, A> {
    fn drop(&mut self) {
        // If the unwrap panics, we will just leak, not destroy, the arena.
        let mut lock = self.owner.0.lock().unwrap();
        /*
         * Safety considerations.
         *
         * The only requirement is that the self.arena is not ever used again. This is trivial, we
         * are in the destructor.
         *
         * We also need to ensure the member is not dropped in here (otherwise we would destroy
         * memory that's still lifetime-OK according to the `'h`. But push doesn't panic
         * (allocators are disallowed from panicking).
         */
        let member = unsafe { ManuallyDrop::take(&mut self.arena) };
        lock.push(member);
    }
}
