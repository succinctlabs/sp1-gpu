use moongate_core::{
    merkle_tree::FieldMerkleTreeDeviceCommitter,
    poseidon2::{baby_bear::DeviceHasherBabyBear, bn254::DeviceHasherBn254},
    stark::StarkGpuProver,
};

use sp1_core_machine::riscv::RiscvAir;
use sp1_prover::{
    components::SP1ProverComponents, CompressAir, CoreSC, InnerSC, OuterSC, ShrinkAir, WrapAir,
};
use sp1_stark::StarkGenericConfig;

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
    type WrapProver = StarkGpuProver<
        OuterSC,
        FieldMerkleTreeDeviceCommitter<DeviceHasherBn254>,
        WrapAir<<OuterSC as StarkGenericConfig>::Val>,
    >;
}
