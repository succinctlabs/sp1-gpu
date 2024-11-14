use std::{
    alloc::Layout,
    ffi::c_void,
    mem,
    ops::Deref,
    ptr::{self, NonNull},
    sync::Arc,
    time::Duration,
};

use moongate_bloc::{
    alloc::{AllocError, Allocator, DeviceMemory},
    bump::Bump,
    herd::Member,
};

use crate::{
    device::{error::CudaError, memory::GlobalDeviceAllocator},
    time::CudaInstant,
};

use super::{event::CudaEvent, ffi, CudaSync};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CudaStreamHandle(*mut c_void);

#[derive(Debug)]
#[repr(transparent)]
pub struct CudaStreamOwned(CudaStreamHandle);

unsafe impl Send for CudaStreamOwned {}
unsafe impl Sync for CudaStreamOwned {}

#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct CudaStream(Arc<CudaStreamOwned>);

/// An adapter that takes any allocator for device memory and performs all the operations with
/// respect to a given stream.
#[derive(Debug, Clone)]
pub struct StreamAllocator<A: Allocator> {
    alloc: A,
    stream: CudaStream,
}

pub type BumpStream = StreamAllocator<Bump<GlobalDeviceAllocator>>;

pub type StreamMember<'h> = StreamAllocator<Member<'h, Bump<GlobalDeviceAllocator>>>;

impl CudaStream {
    pub fn create() -> Result<Self, CudaError> {
        let mut ptr = CudaStreamHandle(ptr::null_mut());
        unsafe { ffi::cuda_stream_create(&mut ptr as *mut CudaStreamHandle) }.to_result()?;
        Ok(Self(Arc::new(CudaStreamOwned(ptr))))
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { ffi::cuda_stream_synchronize(self.0 .0) }.to_result()
    }

    #[inline]
    pub fn handle(&self) -> CudaStreamHandle {
        self.0 .0
    }

    pub fn now(&self) -> Result<CudaInstant, CudaError> {
        let event = CudaEvent::new()?;
        self.record(&event)?;
        Ok(CudaInstant(event))
    }

    pub fn record(&self, event: &CudaEvent) -> Result<(), CudaError> {
        unsafe { ffi::cuda_event_record(event.handle(), self.0 .0) }.to_result()
    }

    pub fn elapsed(&self, start: &CudaInstant) -> Result<Duration, CudaError> {
        let end = CudaEvent::new()?;
        self.record(&end)?;
        end.synchronize()?;
        let mut ms: f32 = 0.0;
        unsafe { ffi::cuda_event_elapsed_time(&mut ms, start.0.handle(), end.handle()) }
            .to_result()?;

        let s = ms as f64 * 1e-3;
        Ok(Duration::from_secs_f64(s))
    }

    #[inline]
    pub fn wait_event(&self, event: &CudaEvent) -> Result<(), CudaError> {
        unsafe { ffi::cuda_stream_wait_event(self.0 .0, event.handle()) }.to_result()
    }

    /// # Safety
    ///
    /// TODO
    unsafe fn cuda_malloc_async<T: Copy>(&self, size: usize) -> Result<*mut T, CudaError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe {
            ffi::cuda_malloc_async(
                &mut ptr as *mut *mut c_void,
                size * mem::size_of::<T>(),
                self.0 .0,
            )
        }
        .to_result()?;
        Ok(ptr as *mut T)
    }

    /// # Safety
    ///
    /// TODO
    #[inline]
    pub unsafe fn try_alloc<T: Copy>(&self, len: usize) -> Result<*mut T, CudaError> {
        self.cuda_malloc_async(len)
    }

    /// # Safety
    ///
    /// TODO
    #[inline]
    pub unsafe fn free_async<T: Copy>(&self, ptr: *mut T) -> Result<(), CudaError> {
        unsafe { ffi::cuda_free_async(ptr as *mut c_void, self.0 .0) }.to_result()
    }

    /// # Safety
    ///
    /// TODO
    #[inline]
    pub unsafe fn mem_set_async<T: Copy>(
        &self,
        dst: *mut T,
        value: u8,
        count: usize,
    ) -> Result<(), CudaError> {
        ffi::cuda_mem_set_async(dst as *mut c_void, value, count * mem::size_of::<T>(), self.0 .0)
            .to_result()
    }

    /// # Safety
    ///
    /// TODO
    pub unsafe fn cuda_memcpy_device_to_device_async<T: Copy>(
        &self,
        dst: *mut T,
        src: *const T,
        count: usize,
    ) -> Result<(), CudaError> {
        unsafe {
            ffi::cuda_mem_copy_device_to_device_async(
                dst as *mut c_void,
                src as *const c_void,
                count * mem::size_of::<T>(),
                self.0 .0,
            )
        }
        .to_result()
    }

    /// # Safety
    ///
    /// TODO
    pub unsafe fn cuda_memcpy_host_to_device_async<T: Copy>(
        &self,
        dst: *mut T,
        src: *const T,
        count: usize,
    ) -> Result<(), CudaError> {
        unsafe {
            ffi::cuda_mem_copy_host_to_device_async(
                dst as *mut c_void,
                src as *const c_void,
                count * mem::size_of::<T>(),
                self.0 .0,
            )
        }
        .to_result()
    }

    /// # Safety
    ///
    /// TODO
    #[inline]
    pub unsafe fn cuda_memcpy_device_to_host_async<T: Copy>(
        &self,
        dst: *mut T,
        src: *const T,
        count: usize,
    ) -> Result<(), CudaError> {
        unsafe {
            ffi::cuda_mem_copy_device_to_host_async(
                dst as *mut c_void,
                src as *const c_void,
                count * mem::size_of::<T>(),
                self.0 .0,
            )
        }
        .to_result()
    }

    /// # Safety
    ///
    /// TODO
    #[inline]
    pub unsafe fn cuda_memcpy_host_to_host_async<T: Copy>(
        &self,
        dst: *mut T,
        src: *const T,
        count: usize,
    ) -> Result<(), CudaError> {
        unsafe {
            ffi::cuda_mem_copy_host_to_host_async(
                dst as *mut c_void,
                src as *const c_void,
                count * mem::size_of::<T>(),
                self.0 .0,
            )
        }
        .to_result()
    }
}

impl Default for CudaStream {
    #[inline]
    fn default() -> Self {
        let raw = CudaStreamOwned(unsafe { ffi::DEFAULT_STREAM });
        Self(Arc::new(raw))
    }
}

impl Drop for CudaStreamOwned {
    fn drop(&mut self) {
        if self.0 != unsafe { ffi::DEFAULT_STREAM } {
            unsafe { ffi::cuda_stream_destroy(self.0) }.to_result().unwrap();
        }
    }
}

impl Deref for CudaStream {
    type Target = CudaStreamOwned;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl Allocator for CudaStream {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let len = layout.size();
            let ptr = self.cuda_malloc_async::<u8>(len).map_err(|_| AllocError)?;
            Ok(NonNull::slice_from_raw_parts(NonNull::new_unchecked(ptr), len))
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        self.free_async(ptr.as_ptr()).unwrap()
    }
}

impl DeviceMemory for CudaStream {
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
    ) -> Result<(), AllocError> {
        self.cuda_memcpy_device_to_device_async::<u8>(dst, src, size).map_err(|_| AllocError)
    }

    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), AllocError> {
        self.mem_set_async(dst, value, size).map_err(|_| AllocError)
    }
}

impl<A: Allocator> StreamAllocator<A> {
    pub const fn new(alloc: A, stream: CudaStream) -> Self {
        Self { alloc, stream }
    }
}

impl CudaSync for CudaStream {
    fn stream(&self) -> &CudaStream {
        self
    }
}

impl<A: Allocator> CudaSync for StreamAllocator<A> {
    fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

impl<A: Allocator> DeviceMemory for StreamAllocator<A> {
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
    ) -> Result<(), AllocError> {
        self.stream.copy_nonoverlapping(src, dst, size)
    }

    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), AllocError> {
        self.stream.write_bytes(dst, value, size)
    }
}

unsafe impl<A: Allocator> Allocator for StreamAllocator<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.alloc.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.alloc.deallocate(ptr, layout);
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::{thread_rng, Rng};

    use crate::{
        device::{
            memory::{ToDeviceIn, ToHost},
            DeviceBuffer,
        },
        utils::init_tracer,
    };

    use super::*;

    #[test]
    fn test_default_stream() {
        let stream = CudaStream::default();
        let event = CudaEvent::new().unwrap();
        stream.record(&event).unwrap();

        // Get a big buffer and measure the time it takes to copy it.
        let data = vec![0u32; 1 << 22];
        let mut buffer = DeviceBuffer::<u32>::with_capacity(data.len()).unwrap();
        let time = stream.now().unwrap();
        buffer.extend_from_host_slice(&data);
        let elapsed = stream.elapsed(&time).unwrap();
        println!("{:?}", elapsed);
        stream.synchronize().unwrap();
    }

    #[test]
    fn test_streams() {
        let stream = CudaStream::create().unwrap();

        // Get a big buffer and measure the time it takes to copy it.
        let data = vec![0u32; 1 << 22];
        let time = stream.now().unwrap();
        unsafe {
            let buf = stream.cuda_malloc_async::<u32>(data.len()).unwrap();
            stream.cuda_memcpy_host_to_device_async(buf, data.as_ptr(), data.len()).unwrap();
            stream.free_async(buf).unwrap();
            let end = CudaEvent::new().unwrap();
            stream.record(&end).unwrap();
            let elapsed = stream.elapsed(&time).unwrap();
            println!("{:?}", elapsed);
        }
    }

    #[test]
    fn test_stream_allocator() {
        let stream = CudaStream::create().unwrap();
        let bump = Bump::<GlobalDeviceAllocator>::default();
        let alloc = StreamAllocator::new(bump, stream);

        let mut buffer = DeviceBuffer::<u8, _>::with_capacity_in(1 << 22, &alloc).unwrap();
        unsafe {
            buffer.set_max_len();
        }
        buffer.set(121).unwrap();
        let host = buffer.to_host();
        for val in host {
            assert_eq!(val, 121);
        }
    }

    #[test]
    #[ignore]
    fn test_release_api() {
        init_tracer();
        let mut rng = thread_rng();

        let heights = [21, 21, 19, 16];
        let widths = [200, 30, 50, 10];

        let host_matrices = heights
            .into_iter()
            .zip_eq(widths)
            .map(|(log_height, width)| {
                let height = 1 << log_height;
                let values = (0..width * height).map(|_| rng.gen::<BabyBear>()).collect::<Vec<_>>();
                RowMajorMatrix::new(values, width)
            })
            .collect::<Vec<_>>();

        // Serial with default stream.
        let mut device_matrices_serial = Vec::with_capacity(host_matrices.len());
        for mat in host_matrices.iter() {
            let mat_span = tracing::debug_span!("serial matrix operation").entered();
            let device_trace = mat.to_device_in(CudaStream::default()).unwrap().to_column_major();
            mat_span.exit();
            device_matrices_serial.push(device_trace);
        }

        // Serial with other streams.
        let mut device_matrices = Vec::with_capacity(host_matrices.len());
        for mat in host_matrices.iter() {
            let stream = CudaStream::create().unwrap();
            let mat_span = tracing::debug_span!("stream serial matrix operation").entered();
            let device_trace = mat.to_device_in(stream.clone()).unwrap().to_column_major();
            mat_span.exit();
            device_matrices.push(device_trace);
        }

        let clone_of_host = host_matrices.clone();
        // Parallel with other streams.
        let mut device_matrices_rx = Vec::with_capacity(host_matrices.len());
        for mat in clone_of_host {
            let stream = CudaStream::create().unwrap();
            let (tx, rx) = oneshot::channel();
            rayon::spawn(move || {
                let mat_span = tracing::debug_span!("stream serial matrix operation").entered();
                let device_trace = mat.to_device_in(stream.clone()).unwrap().to_column_major();
                tx.send(device_trace).unwrap();
                mat_span.exit();
            });
            device_matrices_rx.push(rx);
        }
        let device_matrices_par =
            device_matrices_rx.into_iter().map(|rx| rx.recv().unwrap()).collect::<Vec<_>>();

        let free_on_host = tracing::debug_span!("free host traces").entered();
        drop(host_matrices);
        free_on_host.exit();

        let def_device_span = tracing::debug_span!("free default device traces").entered();
        let stream = CudaStream::default();
        drop(device_matrices_serial);
        stream.synchronize().unwrap();
        def_device_span.exit();

        let def_device_span = tracing::debug_span!("free stream device traces").entered();
        for mat in device_matrices {
            let stream = mat.stream().clone();
            drop(mat);
            stream.synchronize().unwrap();
        }
        def_device_span.exit();

        let def_device_span = tracing::debug_span!("free stream device traces").entered();
        for mat in device_matrices_par {
            let stream = mat.stream().clone();
            drop(mat);
            stream.synchronize().unwrap();
        }
        def_device_span.exit();
    }
}
