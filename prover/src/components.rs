use moongate_core::stark::StarkGpuProver;
use moongate_core::{
    merkle_tree::FieldMerkleTreeDeviceCommitter, poseidon2::baby_bear::DeviceHasherBabyBear,
};
use sp1_core::stark::{CpuProver, RiscvAir, StarkGenericConfig};

use sp1_prover::{
    components::SP1ProverComponents, CompressAir, CoreSC, InnerSC, OuterSC, ShrinkAir, WrapAir,
};

#[derive(Debug, Clone, Copy)]
pub struct GpuProverComponents;

impl SP1ProverComponents for GpuProverComponents {
    type CoreProver = StarkGpuProver<
        CoreSC,
        FieldMerkleTreeDeviceCommitter<DeviceHasherBabyBear>,
        RiscvAir<<CoreSC as StarkGenericConfig>::Val>,
    >;
    type CompressProver = StarkGpuProver<
        InnerSC,
        FieldMerkleTreeDeviceCommitter<DeviceHasherBabyBear>,
        CompressAir<<InnerSC as StarkGenericConfig>::Val>,
    >;
    type ShrinkProver = StarkGpuProver<
        InnerSC,
        FieldMerkleTreeDeviceCommitter<DeviceHasherBabyBear>,
        ShrinkAir<<InnerSC as StarkGenericConfig>::Val>,
    >;
    type WrapProver = CpuProver<OuterSC, WrapAir<<OuterSC as StarkGenericConfig>::Val>>;
}
