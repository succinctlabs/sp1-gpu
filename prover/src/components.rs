use moongate_core::poseidon2::baby_bear::DeviceHasherBabyBear;
use moongate_core::stark::StarkGpuProver;
use sp1_core::stark::{CpuProver, RiscvAir, StarkGenericConfig};

use sp1_prover::{
    components::SP1ProverComponents, CompressAir, CoreSC, InnerSC, OuterSC, ShrinkAir, WrapAir,
};

#[derive(Debug, Clone, Copy)]
pub struct GpuProverComponents;

impl SP1ProverComponents for GpuProverComponents {
    type CoreProver =
        StarkGpuProver<CoreSC, DeviceHasherBabyBear, RiscvAir<<CoreSC as StarkGenericConfig>::Val>>;
    type CompressProver = StarkGpuProver<
        InnerSC,
        DeviceHasherBabyBear,
        CompressAir<<InnerSC as StarkGenericConfig>::Val>,
    >;
    type ShrinkProver = CpuProver<InnerSC, ShrinkAir<<InnerSC as StarkGenericConfig>::Val>>;
    type WrapProver = CpuProver<OuterSC, WrapAir<<OuterSC as StarkGenericConfig>::Val>>;
}
