use std::os::raw::c_void;
use tracing::{span, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

pub struct NvtxLayer;

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for NvtxLayer {
    fn on_new_span(&self, _attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).expect("Failed to get span");

        // Attach start domain to the span
        let domain = NvtxDomainHandle::new(span.name());
        span.extensions_mut().insert(domain);
    }

    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).expect("Failed to get span");

        // Retrieve start time and calculate the duration
        let extensions = span.extensions();
        if let Some(domain) = extensions.get::<NvtxDomainHandle>() {
            unsafe { ffi::nvtx_domain_destroy(*domain) }
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
struct NvtxDomainHandle(*mut c_void);

unsafe impl Send for NvtxDomainHandle {}
unsafe impl Sync for NvtxDomainHandle {}

mod ffi {
    use std::ffi::c_char;

    use super::NvtxDomainHandle;

    extern "C" {
        #[link_name = "nvtxDomainCreateARust"]
        pub fn nvtx_domain_create(name: *const c_char) -> NvtxDomainHandle;

        #[link_name = "nvtxDomainDestroyARust"]
        pub fn nvtx_domain_destroy(domain: NvtxDomainHandle);
    }
}

impl NvtxDomainHandle {
    pub fn new(name: &str) -> Self {
        let name = std::ffi::CString::new(name).unwrap();
        unsafe { ffi::nvtx_domain_create(name.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvtx_domain() {
        let domain = NvtxDomainHandle::new("test");
        unsafe { ffi::nvtx_domain_destroy(domain) };
    }
}
