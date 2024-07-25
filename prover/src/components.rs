use moongate_core::stark::StarkGpuProver;
use sp1_core::stark::{CpuProver, RiscvAir, StarkGenericConfig};

use sp1_prover::{
    components::SP1ProverComponents, CompressAir, CoreSC, InnerSC, OuterSC, ShrinkAir, WrapAir,
};

#[derive(Debug, Clone, Copy)]
pub struct GpuProverComponents;

impl SP1ProverComponents for GpuProverComponents {
    type CoreProver = StarkGpuProver<CoreSC, RiscvAir<<CoreSC as StarkGenericConfig>::Val>>;
    type CompressProver =
        StarkGpuProver<InnerSC, CompressAir<<InnerSC as StarkGenericConfig>::Val>>;
    type ShrinkProver = CpuProver<InnerSC, ShrinkAir<<InnerSC as StarkGenericConfig>::Val>>;
    type WrapProver = CpuProver<OuterSC, WrapAir<<OuterSC as StarkGenericConfig>::Val>>;
}
