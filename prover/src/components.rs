use moongate_core::stark::StarkGpuProver;
use sp1_core::stark::{DefaultProver, RiscvAir, StarkGenericConfig};

use sp1_prover::{
    components::SP1ProverComponents, CompressAir, CoreSC, InnerSC, OuterSC, ReduceAir, WrapAir,
};

#[derive(Debug, Clone, Copy)]
pub struct GpuProverComponents;

impl SP1ProverComponents for GpuProverComponents {
    type CoreProver = StarkGpuProver<CoreSC, RiscvAir<<CoreSC as StarkGenericConfig>::Val>>;
    type CompressProver = StarkGpuProver<InnerSC, ReduceAir<<InnerSC as StarkGenericConfig>::Val>>;
    type ShrinkProver = DefaultProver<InnerSC, CompressAir<<InnerSC as StarkGenericConfig>::Val>>;
    type WrapProver = DefaultProver<OuterSC, WrapAir<<OuterSC as StarkGenericConfig>::Val>>;
}
