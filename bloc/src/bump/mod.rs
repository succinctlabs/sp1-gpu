//! See https://github.com/fitzgen/bumpalo/tree/main

use core::{
    alloc::Layout,
    cell::Cell,
    iter,
    marker::PhantomData,
    mem,
    ptr::{self, NonNull},
    slice,
};

use crate::alloc::{AllocError, Allocator};

#[derive(Debug)]
pub struct Bump<A: Allocator> {
    current_chunk_footer: Cell<NonNull<ChunkFooter>>,
    allocation_limit: Cell<Option<usize>>,
    pool_alloc: A,
}

#[repr(C)]
#[derive(Debug)]
struct ChunkFooter {
    // Pointer to the start of this chunk allocation. This footer is always at
    // the end of the chunk.
    data: NonNull<u8>,

    // The layout of this chunk's allocation.
    layout: Layout,

    // Link to the previous chunk.
    //
    // Note that the last node in the `prev` linked list is the canonical empty
    // chunk, whose `prev` link points to itself.
    prev: Cell<NonNull<ChunkFooter>>,

    // Bump allocation finger that is always in the range `self.data..=self`.
    ptr: Cell<NonNull<u8>>,

    // The bytes allocated in all chunks so far, the canonical empty chunk has
    // a size of 0 and for all other chunks, `allocated_bytes` will be
    // the allocated_bytes of the current chunk plus the allocated bytes
    // of the `prev` chunk.
    allocated_bytes: usize,
}

/// A wrapper type for the canonical, statically allocated empty chunk.
///
/// For the canonical empty chunk to be `static`, its type must be `Sync`, which
/// is the purpose of this wrapper type. This is safe because the empty chunk is
/// immutable and never actually modified.
#[repr(transparent)]
struct EmptyChunkFooter(ChunkFooter);

unsafe impl Sync for EmptyChunkFooter {}

static EMPTY_CHUNK: EmptyChunkFooter = EmptyChunkFooter(ChunkFooter {
    // This chunk is empty (except the foot itself).
    layout: Layout::new::<ChunkFooter>(),

    // The start of the (empty) allocatable region for this chunk is itself.
    data: unsafe { NonNull::new_unchecked(&EMPTY_CHUNK as *const EmptyChunkFooter as *mut u8) },

    // The end of the (empty) allocatable region for this chunk is also itself.
    ptr: Cell::new(unsafe {
        NonNull::new_unchecked(&EMPTY_CHUNK as *const EmptyChunkFooter as *mut u8)
    }),

    // Invariant: the last chunk footer in all `ChunkFooter::prev` linked lists
    // is the empty chunk footer, whose `prev` points to itself.
    prev: Cell::new(unsafe {
        NonNull::new_unchecked(&EMPTY_CHUNK as *const EmptyChunkFooter as *mut ChunkFooter)
    }),

    // Empty chunks count as 0 allocated bytes in an arena.
    allocated_bytes: 0,
});

impl EmptyChunkFooter {
    fn get(&'static self) -> NonNull<ChunkFooter> {
        NonNull::from(&self.0)
    }
}

impl ChunkFooter {
    // Returns the start and length of the currently allocated region of this
    // chunk.
    fn as_raw_parts(&self) -> (*const u8, usize) {
        let data = self.data.as_ptr() as *const u8;
        let ptr = self.ptr.get().as_ptr() as *const u8;
        debug_assert!(data <= ptr);
        debug_assert!(ptr <= self as *const ChunkFooter as *const u8);
        let len = unsafe { (self as *const ChunkFooter as *const u8).offset_from(ptr) as usize };
        (ptr, len)
    }

    /// Is this chunk the last empty chunk?
    fn is_empty(&self) -> bool {
        ptr::eq(self, EMPTY_CHUNK.get().as_ptr())
    }
}

impl<A: Allocator> Drop for Bump<A> {
    fn drop(&mut self) {
        unsafe {
            dealloc_chunk_list(self.current_chunk_footer.get(), &self.pool_alloc);
        }
    }
}

#[inline]
fn is_pointer_aligned_to<T>(pointer: *mut T, align: usize) -> bool {
    debug_assert!(align.is_power_of_two());

    let pointer = pointer as usize;
    let pointer_aligned = round_down_to(pointer, align);
    pointer == pointer_aligned
}

#[inline]
pub(crate) fn round_up_to(n: usize, divisor: usize) -> Option<usize> {
    debug_assert!(divisor > 0);
    debug_assert!(divisor.is_power_of_two());
    Some(n.checked_add(divisor - 1)? & !(divisor - 1))
}

#[inline]
pub(crate) fn round_down_to(n: usize, divisor: usize) -> usize {
    debug_assert!(divisor > 0);
    debug_assert!(divisor.is_power_of_two());
    n & !(divisor - 1)
}

/// Same as `round_down_to` but preserves pointer provenance.
#[inline]
pub(crate) fn round_mut_ptr_down_to(ptr: *mut u8, divisor: usize) -> *mut u8 {
    debug_assert!(divisor > 0);
    debug_assert!(divisor.is_power_of_two());
    ptr.wrapping_sub(ptr as usize & (divisor - 1))
}

// `Bump`s are safe to send between threads because nothing aliases its owned
// chunks until you start allocating from it. But by the time you allocate from
// it, the returned references to allocations borrow the `Bump` and therefore
// prevent sending the `Bump` across threads until the borrows end.
unsafe impl<A: Allocator + Send> Send for Bump<A> {}

#[inline]
unsafe fn dealloc_chunk_list<A: Allocator>(mut footer: NonNull<ChunkFooter>, alloc: &A) {
    while !footer.as_ref().is_empty() {
        let f = footer;
        footer = f.as_ref().prev.get();
        alloc.deallocate(f.as_ref().data, f.as_ref().layout);
    }
}

// After this point, we try to hit page boundaries instead of powers of 2
const PAGE_STRATEGY_CUTOFF: usize = 0x1000;

// We only support alignments of up to 16 bytes for iter_allocated_chunks.
const SUPPORTED_ITER_ALIGNMENT: usize = 16;
const CHUNK_ALIGN: usize = SUPPORTED_ITER_ALIGNMENT;
const FOOTER_SIZE: usize = mem::size_of::<ChunkFooter>();

// Assert that ChunkFooter is at most the supported alignment. This will give a compile time error if it is not the case
const _FOOTER_ALIGN_ASSERTION: bool = mem::align_of::<ChunkFooter>() <= CHUNK_ALIGN;
const _: [(); _FOOTER_ALIGN_ASSERTION as usize] = [()];

// Maximum typical overhead per allocation imposed by allocators.
const MALLOC_OVERHEAD: usize = 16;

// This is the overhead from malloc, footer and alignment. For instance, if
// we want to request a chunk of memory that has at least X bytes usable for
// allocations (where X is aligned to CHUNK_ALIGN), then we expect that the
// after adding a footer, malloc overhead and alignment, the chunk of memory
// the allocator actually sets aside for us is X+OVERHEAD rounded up to the
// nearest suitable size boundary.
const OVERHEAD: usize = (MALLOC_OVERHEAD + FOOTER_SIZE + (CHUNK_ALIGN - 1)) & !(CHUNK_ALIGN - 1);

// Choose a relatively small default initial chunk size, since we double chunk
// sizes as we grow bump arenas to amortize costs of hitting the global
// allocator.
const FIRST_ALLOCATION_GOAL: usize = 1 << 9;

// The actual size of the first allocation is going to be a bit smaller
// than the goal. We need to make room for the footer, and we also need
// take the alignment into account.
const DEFAULT_CHUNK_SIZE_WITHOUT_FOOTER: usize = FIRST_ALLOCATION_GOAL - OVERHEAD;

/// The memory size and alignment details for a potential new chunk
/// allocation.
#[derive(Debug, Clone, Copy)]
struct NewChunkMemoryDetails {
    new_size_without_footer: usize,
    align: usize,
    size: usize,
}

/// Wrapper around `Layout::from_size_align` that adds debug assertions.
#[inline]
fn layout_from_size_align(size: usize, align: usize) -> Result<Layout, AllocError> {
    Layout::from_size_align(size, align).map_err(|_| AllocError)
}

#[inline(never)]
fn allocation_size_overflow<T>() -> T {
    panic!("requested allocation size overflowed")
}

impl<A: Allocator> Bump<A> {
    /// Construct a new arena with the specified byte capacity to bump allocate into.
    pub fn with_capacity_in(capacity: usize, pool_alloc: A) -> Self {
        Self::try_with_capacity_in(capacity, pool_alloc).unwrap_or_else(|_| oom())
    }

    /// Attempt to construct a new arena with the specified byte capacity to bump allocate into.
    pub fn try_with_capacity_in(capacity: usize, pool_alloc: A) -> Result<Self, AllocError> {
        if capacity == 0 {
            return Ok(Self {
                current_chunk_footer: Cell::new(EMPTY_CHUNK.get()),
                allocation_limit: Cell::new(None),
                pool_alloc,
            });
        }

        let layout = layout_from_size_align(capacity, 1)?;

        let chunk_footer = unsafe {
            Self::new_chunk(
                Self::new_chunk_memory_details(None, layout).ok_or(AllocError)?,
                layout,
                EMPTY_CHUNK.get(),
                &pool_alloc,
            )
            .ok_or(AllocError)?
        };

        Ok(Self {
            current_chunk_footer: Cell::new(chunk_footer),
            allocation_limit: Cell::new(None),
            pool_alloc,
        })
    }

    /// Allocate a new chunk and return its initialized footer.
    ///
    /// If given, `layouts` is a tuple of the current chunk size and the
    /// layout of the allocation request that triggered us to fall back to
    /// allocating a new chunk of memory.
    unsafe fn new_chunk(
        new_chunk_memory_details: NewChunkMemoryDetails,
        requested_layout: Layout,
        prev: NonNull<ChunkFooter>,
        alloc: &A,
    ) -> Option<NonNull<ChunkFooter>> {
        let NewChunkMemoryDetails { new_size_without_footer, align, size } =
            new_chunk_memory_details;

        let layout = layout_from_size_align(size, align).ok()?;

        debug_assert!(size >= requested_layout.size());

        // Try to allocate and return `None` on failure.
        let data = alloc.allocate(layout).ok()?;

        // TODO: check the allocation length and perhaps give extra space.
        // let alloc_len = data.len();

        // The `ChunkFooter` is at the end of the chunk.
        let footer_ptr = data.cast::<u8>().as_ptr().add(new_size_without_footer);
        debug_assert_eq!((data.cast::<u8>().as_ptr() as usize) % align, 0);
        debug_assert_eq!(footer_ptr as usize % CHUNK_ALIGN, 0);
        let footer_ptr = footer_ptr as *mut ChunkFooter;

        // The bump pointer is initialized to the end of the range we will
        // bump out of.
        let ptr = Cell::new(NonNull::new_unchecked(footer_ptr as *mut u8));

        // The `allocated_bytes` of a new chunk counts the total size
        // of the chunks, not how much of the chunks are used.
        let allocated_bytes = prev.as_ref().allocated_bytes + new_size_without_footer;

        ptr::write(
            footer_ptr,
            ChunkFooter { data: data.cast(), layout, prev: Cell::new(prev), ptr, allocated_bytes },
        );

        Some(NonNull::new_unchecked(footer_ptr))
    }

    /// The allocation limit for this arena in bytes.
    pub fn allocation_limit(&self) -> Option<usize> {
        self.allocation_limit.get()
    }

    /// Set the allocation limit in bytes for this arena.
    ///
    /// The allocation limit is only enforced when allocating new backing chunks for
    /// a `Bump`. Updating the allocation limit will not affect existing allocations
    /// or any future allocations within the `Bump`'s current chunk.
    ///
    /// ## Example
    ///
    /// ```
    /// let bump = bumpalo::Bump::with_capacity(0);
    ///
    /// bump.set_allocation_limit(Some(0));
    ///
    /// assert!(bump.try_alloc(5).is_err());
    /// ```
    pub fn set_allocation_limit(&self, limit: Option<usize>) {
        self.allocation_limit.set(limit);
    }

    /// How much headroom an arena has before it hits its allocation
    /// limit.
    fn allocation_limit_remaining(&self) -> Option<usize> {
        self.allocation_limit.get().and_then(|allocation_limit| {
            let allocated_bytes = self.allocated_bytes();
            if allocated_bytes > allocation_limit {
                None
            } else {
                Some(usize::abs_diff(allocation_limit, allocated_bytes))
            }
        })
    }

    /// Calculates the number of bytes currently allocated across all chunks in
    /// this bump arena.
    ///
    /// If you allocate types of different alignments or types with
    /// larger-than-typical alignment in the same arena, some padding
    /// bytes might get allocated in the bump arena. Note that those padding
    /// bytes will add to this method's resulting sum, so you cannot rely
    /// on it only counting the sum of the sizes of the things
    /// you've allocated in the arena.
    ///
    /// The allocated bytes do not include the size of bumpalo's metadata,
    /// so the amount of memory requested from the Rust allocator is higher
    /// than the returned value.
    ///
    /// ## Example
    ///
    /// ```
    /// let bump = bumpalo::Bump::new();
    /// let _x = bump.alloc_slice_fill_default::<u32>(5);
    /// let bytes = bump.allocated_bytes();
    /// assert!(bytes >= core::mem::size_of::<u32>() * 5);
    /// ```
    pub fn allocated_bytes(&self) -> usize {
        let footer = self.current_chunk_footer.get();

        unsafe { footer.as_ref().allocated_bytes }
    }

    /// Whether a request to allocate a new chunk with a given size for a given
    /// requested layout will fit under the allocation limit set on a `Bump`.
    fn chunk_fits_under_limit(
        allocation_limit_remaining: Option<usize>,
        new_chunk_memory_details: NewChunkMemoryDetails,
    ) -> bool {
        allocation_limit_remaining
            .map(|allocation_limit_left| {
                allocation_limit_left >= new_chunk_memory_details.new_size_without_footer
            })
            .unwrap_or(true)
    }

    /// Determine the memory details including final size, alignment and
    /// final size without footer for a new chunk that would be allocated
    /// to fulfill an allocation request.
    fn new_chunk_memory_details(
        new_size_without_footer: Option<usize>,
        requested_layout: Layout,
    ) -> Option<NewChunkMemoryDetails> {
        let mut new_size_without_footer =
            new_size_without_footer.unwrap_or(DEFAULT_CHUNK_SIZE_WITHOUT_FOOTER);

        // We want to have CHUNK_ALIGN or better alignment
        let mut align = CHUNK_ALIGN;

        // If we already know we need to fulfill some request,
        // make sure we allocate at least enough to satisfy it
        align = align.max(requested_layout.align());
        let requested_size =
            round_up_to(requested_layout.size(), align).unwrap_or_else(allocation_size_overflow);
        new_size_without_footer = new_size_without_footer.max(requested_size);

        // We want our allocations to play nice with the memory allocator,
        // and waste as little memory as possible.
        // For small allocations, this means that the entire allocation
        // including the chunk footer and mallocs internal overhead is
        // as close to a power of two as we can go without going over.
        // For larger allocations, we only need to get close to a page
        // boundary without going over.
        if new_size_without_footer < PAGE_STRATEGY_CUTOFF {
            new_size_without_footer =
                (new_size_without_footer + OVERHEAD).next_power_of_two() - OVERHEAD;
        } else {
            new_size_without_footer =
                round_up_to(new_size_without_footer + OVERHEAD, 0x1000)? - OVERHEAD;
        }

        debug_assert_eq!(align % CHUNK_ALIGN, 0);
        debug_assert_eq!(new_size_without_footer % CHUNK_ALIGN, 0);
        let size = new_size_without_footer
            .checked_add(FOOTER_SIZE)
            .unwrap_or_else(allocation_size_overflow);

        Some(NewChunkMemoryDetails { new_size_without_footer, size, align })
    }

    /// Reset this bump allocator.
    ///
    /// Performs mass deallocation on everything allocated in this arena by
    /// resetting the pointer into the underlying chunk of memory to the start
    /// of the chunk. Does not run any `Drop` implementations on deallocated
    /// objects; see [the top-level documentation](struct.Bump.html) for details.
    ///
    /// If this arena has allocated multiple chunks to bump allocate into, then
    /// the excess chunks are returned to the global allocator.
    pub fn reset(&mut self) {
        // Takes `&mut self` so `self` must be unique and there can't be any
        // borrows active that would get invalidated by resetting.
        unsafe {
            if self.current_chunk_footer.get().as_ref().is_empty() {
                return;
            }

            let mut cur_chunk = self.current_chunk_footer.get();

            // Deallocate all chunks except the current one
            let prev_chunk = cur_chunk.as_ref().prev.replace(EMPTY_CHUNK.get());
            dealloc_chunk_list(prev_chunk, &self.pool_alloc);

            // Reset the bump finger to the end of the chunk.
            cur_chunk.as_ref().ptr.set(cur_chunk.cast());

            // Reset the allocated size of the chunk.
            cur_chunk.as_mut().allocated_bytes = cur_chunk.as_ref().layout.size();

            debug_assert!(
                self.current_chunk_footer.get().as_ref().prev.get().as_ref().is_empty(),
                "We should only have a single chunk"
            );
            debug_assert_eq!(
                self.current_chunk_footer.get().as_ref().ptr.get(),
                self.current_chunk_footer.get().cast(),
                "Our chunk's bump finger should be reset to the start of its allocation"
            );
        }
    }

    /// Attempts to allocate space for an object with the given `Layout` or else returns
    /// an `Err`.
    ///
    /// The returned pointer points at uninitialized memory, and should be
    /// initialized with
    /// [`std::ptr::write`](https://doc.rust-lang.org/std/ptr/fn.write.html).
    ///
    /// # Errors
    ///
    /// Errors if reserving space matching `layout` fails.
    #[inline(always)]
    pub fn try_alloc_layout(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        if let Some(p) = self.try_alloc_layout_fast(layout) {
            Ok(p)
        } else {
            self.alloc_layout_slow(layout).ok_or(AllocError)
        }
    }

    #[inline(always)]
    fn try_alloc_layout_fast(&self, layout: Layout) -> Option<NonNull<u8>> {
        // We don't need to check for ZSTs here since they will automatically
        // be handled properly: the pointer will be bumped by zero bytes,
        // modulo alignment. This keeps the fast path optimized for non-ZSTs,
        // which are much more common.
        unsafe {
            let footer = self.current_chunk_footer.get();
            let footer = footer.as_ref();
            let ptr = footer.ptr.get().as_ptr();
            let start = footer.data.as_ptr();
            debug_assert!(start <= ptr);
            debug_assert!(ptr as *const u8 <= footer as *const _ as *const u8);

            if (ptr as usize) < layout.size() {
                return None;
            }

            let ptr = ptr.wrapping_sub(layout.size());
            let aligned_ptr = round_mut_ptr_down_to(ptr, layout.align());

            if aligned_ptr >= start {
                let aligned_ptr = NonNull::new_unchecked(aligned_ptr);
                footer.ptr.set(aligned_ptr);
                Some(aligned_ptr)
            } else {
                None
            }
        }
    }

    /// Gets the remaining capacity in the current chunk (in bytes).
    pub fn chunk_capacity(&self) -> usize {
        let current_footer = self.current_chunk_footer.get();
        let current_footer = unsafe { current_footer.as_ref() };

        current_footer.ptr.get().as_ptr() as usize - current_footer.data.as_ptr() as usize
    }

    /// Slow path allocation for when we need to allocate a new chunk from the
    /// parent bump set because there isn't enough room in our current chunk.
    #[inline(never)]
    #[cold]
    fn alloc_layout_slow(&self, layout: Layout) -> Option<NonNull<u8>> {
        unsafe {
            let size = layout.size();
            let allocation_limit_remaining = self.allocation_limit_remaining();

            // Get a new chunk from the global allocator.
            let current_footer = self.current_chunk_footer.get();
            let current_layout = current_footer.as_ref().layout;

            // By default, we want our new chunk to be about twice as big
            // as the previous chunk. If the global allocator refuses it,
            // we try to divide it by half until it works or the requested
            // size is smaller than the default footer size.
            let min_new_chunk_size = layout.size().max(DEFAULT_CHUNK_SIZE_WITHOUT_FOOTER);
            let mut base_size =
                (current_layout.size() - FOOTER_SIZE).checked_mul(2)?.max(min_new_chunk_size);
            let chunk_memory_details = iter::from_fn(|| {
                let bypass_min_chunk_size_for_small_limits = matches!(self.allocation_limit(), Some(limit) if layout.size() < limit
                            && base_size >= layout.size()
                            && limit < DEFAULT_CHUNK_SIZE_WITHOUT_FOOTER
                            && self.allocated_bytes() == 0);

                if base_size >= min_new_chunk_size || bypass_min_chunk_size_for_small_limits {
                    let size = base_size;
                    base_size /= 2;
                    Self::new_chunk_memory_details(Some(size), layout)
                } else {
                    None
                }
            });

            let new_footer = chunk_memory_details
                .filter_map(|chunk_memory_details| {
                    if Self::chunk_fits_under_limit(
                        allocation_limit_remaining,
                        chunk_memory_details,
                    ) {
                        Self::new_chunk(
                            chunk_memory_details,
                            layout,
                            current_footer,
                            &self.pool_alloc,
                        )
                    } else {
                        None
                    }
                })
                .next()?;

            debug_assert_eq!(new_footer.as_ref().data.as_ptr() as usize % layout.align(), 0);

            // Set the new chunk as our new current chunk.
            self.current_chunk_footer.set(new_footer);

            let new_footer = new_footer.as_ref();

            // Move the bump ptr finger down to allocate room for `val`. We know
            // this can't overflow because we successfully allocated a chunk of
            // at least the requested size.
            let mut ptr = new_footer.ptr.get().as_ptr().sub(size);
            // Round the pointer down to the requested alignment.
            ptr = round_mut_ptr_down_to(ptr, layout.align());
            debug_assert!(ptr as *const _ <= new_footer, "{:p} <= {:p}", ptr, new_footer);
            let ptr = NonNull::new_unchecked(ptr);
            new_footer.ptr.set(ptr);

            // Return a pointer to the freshly allocated region in this chunk.
            Some(ptr)
        }
    }

    #[inline]
    unsafe fn is_last_allocation(&self, ptr: NonNull<u8>) -> bool {
        let footer = self.current_chunk_footer.get();
        let footer = footer.as_ref();
        footer.ptr.get() == ptr
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: NonNull<u8>, layout: Layout) {
        // If the pointer is the last allocation we made, we can reuse the bytes,
        // otherwise they are simply leaked -- at least until somebody calls reset().
        if self.is_last_allocation(ptr) {
            let ptr = self.current_chunk_footer.get().as_ref().ptr.get();
            let ptr = NonNull::new_unchecked(ptr.as_ptr().add(layout.size()));
            self.current_chunk_footer.get().as_ref().ptr.set(ptr);
        }
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<u8>, AllocError> {
        // If the new layout demands greater alignment than the old layout has,
        // then either
        //
        // 1. the pointer happens to satisfy the new layout's alignment, so we
        //    got lucky and can return the pointer as-is, or
        //
        // 2. the pointer is not aligned to the new layout's demanded alignment,
        //    and we are unlucky.
        //
        // In the case of (2), to successfully "shrink" the allocation, we would
        // have to allocate a whole new region for the new layout, without being
        // able to free the old region. That is unacceptable, so simply return
        // an allocation failure error instead.
        if old_layout.align() < new_layout.align() {
            if is_pointer_aligned_to(ptr.as_ptr(), new_layout.align()) {
                return Ok(ptr);
            } else {
                return Err(AllocError);
            }
        }

        debug_assert!(is_pointer_aligned_to(ptr.as_ptr(), new_layout.align()));

        let old_size = old_layout.size();
        let new_size = new_layout.size();

        // This is how much space we would *actually* reclaim while satisfying
        // the requested alignment.
        let delta = round_down_to(old_size - new_size, new_layout.align());

        if self.is_last_allocation(ptr)
                // Only reclaim the excess space (which requires a copy) if it
                // is worth it: we are actually going to recover "enough" space
                // and we can do a non-overlapping copy.
                //
                // We do `(old_size + 1) / 2` so division rounds up rather than
                // down. Consider when:
                //
                //     old_size = 5
                //     new_size = 3
                //
                // If we do not take care to round up, this will result in:
                //
                //     delta = 2
                //     (old_size / 2) = (5 / 2) = 2
                //
                // And the the check will succeed even though we are have
                // overlapping ranges:
                //
                //     |--------old-allocation-------|
                //     |------from-------|
                //                 |-------to--------|
                //     +-----+-----+-----+-----+-----+
                //     |  a  |  b  |  c  |  .  |  .  |
                //     +-----+-----+-----+-----+-----+
                //
                // But we MUST NOT have overlapping ranges because we use
                // `copy_nonoverlapping` below! Therefore, we round the division
                // up to avoid this issue.
                && delta >= (old_size + 1) / 2
        {
            let footer = self.current_chunk_footer.get();
            let footer = footer.as_ref();

            // NB: new_ptr is aligned, because ptr *has to* be aligned, and we
            // made sure delta is aligned.
            let new_ptr = NonNull::new_unchecked(footer.ptr.get().as_ptr().add(delta));
            footer.ptr.set(new_ptr);

            // NB: we know it is non-overlapping because of the size check
            // in the `if` condition.
            ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), new_size);

            return Ok(new_ptr);
        }

        // If this wasn't the last allocation, or shrinking wasn't worth it,
        // simply return the old pointer as-is.
        Ok(ptr)
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<u8>, AllocError> {
        let old_size = old_layout.size();
        let new_size = new_layout.size();
        let align_is_compatible = old_layout.align() >= new_layout.align();

        if align_is_compatible && self.is_last_allocation(ptr) {
            // Try to allocate the delta size within this same block so we can
            // reuse the currently allocated space.
            let delta = new_size - old_size;
            if let Some(p) =
                self.try_alloc_layout_fast(layout_from_size_align(delta, old_layout.align())?)
            {
                ptr::copy(ptr.as_ptr(), p.as_ptr(), old_size);
                return Ok(p);
            }
        }

        // Fallback: do a fresh allocation and copy the existing data into it.
        let new_ptr = self.try_alloc_layout(new_layout)?;
        ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), old_size);
        Ok(new_ptr)
    }
}

#[inline(never)]
#[cold]
fn oom() -> ! {
    panic!("out of memory")
}

unsafe impl<'a, A: Allocator> Allocator for &'a Bump<A> {
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.try_alloc_layout(layout)
            .map(|p| unsafe {
                NonNull::new_unchecked(ptr::slice_from_raw_parts_mut(p.as_ptr(), layout.size()))
            })
            .map_err(|_| AllocError)
    }

    #[inline]
    fn allocate_zeroed(&self, _layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unimplemented!()
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        Bump::<A>::dealloc(self, ptr, layout)
    }

    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Bump::<A>::shrink(self, ptr, old_layout, new_layout)
            .map(|p| unsafe {
                NonNull::new_unchecked(ptr::slice_from_raw_parts_mut(p.as_ptr(), new_layout.size()))
            })
            .map_err(|_| AllocError)
    }

    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        Bump::<A>::grow(self, ptr, old_layout, new_layout)
            .map(|p| unsafe {
                NonNull::new_unchecked(ptr::slice_from_raw_parts_mut(p.as_ptr(), new_layout.size()))
            })
            .map_err(|_| AllocError)
    }

    #[inline]
    unsafe fn grow_zeroed(
        &self,
        _ptr: NonNull<u8>,
        _old_layout: Layout,
        _new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unimplemented!()
    }
}

/// An iterator over each chunk of allocated memory that
/// an arena has bump allocated into.
///
/// The chunks are returned ordered by allocation time, with the most recently
/// allocated chunk being returned first.
///
/// The values inside each chunk are also ordered by allocation time, with the most
/// recent allocation being earlier in the slice.
#[derive(Debug)]
pub struct ChunkIter<'a, A: Allocator> {
    raw: ChunkRawIter<'a, A>,
    bump: PhantomData<&'a mut Bump<A>>,
}

impl<'a, A: Allocator> Iterator for ChunkIter<'a, A> {
    type Item = &'a [mem::MaybeUninit<u8>];
    fn next(&mut self) -> Option<&'a [mem::MaybeUninit<u8>]> {
        unsafe {
            let (ptr, len) = self.raw.next()?;
            let slice = slice::from_raw_parts(ptr as *const mem::MaybeUninit<u8>, len);
            Some(slice)
        }
    }
}

impl<'a, A: Allocator> iter::FusedIterator for ChunkIter<'a, A> {}

/// An iterator over raw pointers to chunks of allocated memory that this
/// arena has bump allocated into.
#[derive(Debug)]
pub struct ChunkRawIter<'a, A: Allocator> {
    footer: NonNull<ChunkFooter>,
    bump: PhantomData<&'a Bump<A>>,
}

impl<A: Allocator> Iterator for ChunkRawIter<'_, A> {
    type Item = (*mut u8, usize);
    fn next(&mut self) -> Option<(*mut u8, usize)> {
        unsafe {
            let foot = self.footer.as_ref();
            if foot.is_empty() {
                return None;
            }
            let (ptr, len) = foot.as_raw_parts();
            self.footer = foot.prev.get();
            Some((ptr as *mut u8, len))
        }
    }
}
