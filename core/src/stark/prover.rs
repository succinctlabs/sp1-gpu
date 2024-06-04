use sp1_core::stark::{MachineProof, StarkGenericConfig, StarkMachine, StarkProvingKey};

pub trait GpuProverConfig {
    type SC: StarkGenericConfig;
    type Air;
    type Runtime;
}

pub struct GpuProver<C: GpuProverConfig> {
    machine: StarkMachine<C::SC, C::Air>,
}

impl<C: GpuProverConfig> GpuProver<C> {
    pub fn prove(&self, pk: &StarkProvingKey<C::SC>) -> MachineProof<C::SC> {
        todo!()
    }
}
