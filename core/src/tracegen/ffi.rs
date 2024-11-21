use crate::cuda_runtime::stream::CudaStreamHandle;
use crate::matrix::MatrixViewMutDevice;
use p3_baby_bear::BabyBear;
use sp1_core_executor::events::AluEvent;

pub mod add_sub {
    use super::*;

    /// cbindgen:ignore
    #[allow(unused_attributes)]
    #[link_name = "moongate::add_sub"]
    extern "C" {
        pub fn generate_trace(
            trace: MatrixViewMutDevice<BabyBear>,
            events: *const AluEvent,
            nb_events: u32,
            stream: CudaStreamHandle,
        );
    }
}
