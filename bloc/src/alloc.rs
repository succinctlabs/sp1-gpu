use core::{alloc::Layout, ptr::NonNull};

use thiserror::Error;

#[derive(Copy, Clone, PartialEq, Eq, Debug, Error)]
#[error("allocation error")]
pub struct AllocError;

/// An implementation of `Allocator` can allocate, grow, shrink, and deallocate arbitrary blocks of
/// data described via [`Layout`][]
///
/// # Safety
///
/// * Memory blocks returned from an allocator that are [*currently allocated*] must point to
///   valid memory and retain their validity while they are [*currently allocated*] and the shorter
///   of:
///   - the borrow-checker lifetime of the allocator type itself.
///   - as long as at least one of the instance and all of its clones has not been dropped.
///
/// * copying, cloning, or moving the allocator must not invalidate memory blocks returned from this
///   allocator. A copied or cloned allocator must behave like the same allocator, and
///
/// * any pointer to a memory block which is [*currently allocated*] may be passed to any other
///   method of the allocator.
pub unsafe trait Allocator {
    // Required methods
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;

    #[allow(clippy::missing_safety_doc)]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;

    #[allow(clippy::missing_safety_doc)]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError>;

    #[allow(clippy::missing_safety_doc)]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError>;

    #[allow(clippy::missing_safety_doc)]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError>;

    fn by_ref(&self) -> &Self
    where
        Self: Sized,
    {
        self
    }
}

unsafe impl<A> Allocator for &A
where
    A: Allocator + ?Sized,
{
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate(layout)
    }

    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        (**self).allocate_zeroed(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).deallocate(ptr, layout) }
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).grow(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).grow_zeroed(ptr, old_layout, new_layout) }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the safety contract must be upheld by the caller
        unsafe { (**self).shrink(ptr, old_layout, new_layout) }
    }
}
