use super::{BabyBearFriConfig, StarkGpuProver};

pub struct SyncProver<SC: BabyBearFriConfig, C, A> {
    inner: StarkGpuProver<SC, C, A>,
}
