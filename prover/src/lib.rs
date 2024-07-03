use components::GpuProverComponents;
use sp1_prover::SP1Prover;

pub mod components;

pub type SP1GpuProver = SP1Prover<GpuProverComponents>;

#[cfg(test)]
mod tests {
    use moongate_core::utils::init_tracer;
    use sp1_core::{runtime::SP1Context, utils::SP1ProverOpts};
    use sp1_prover::{tests::test_e2e_prover, SP1Stdin};

    use crate::{components::GpuProverComponents, SP1GpuProver};

    pub const TENDERMINT_BENCHMARK_ELF: &[u8] =
        include_bytes!("../../tendermint_benchmark/elf/riscv32im-succinct-zkvm-elf");

    #[test]
    fn test_e2e_fibonacci() {
        init_tracer();
        test_e2e_prover::<GpuProverComponents>().unwrap()
    }

    #[test]
    #[ignore]
    fn test_core_tendermint() {
        let elf = TENDERMINT_BENCHMARK_ELF;

        init_tracer();

        tracing::info!("Initialzing prover");
        let prover = SP1GpuProver::new();

        let opts = SP1ProverOpts::default();
        let context = SP1Context::default();

        tracing::info!("setup elf");
        let (pk, vk) = prover.setup(elf);

        tracing::info!("prove core");
        let stdin = SP1Stdin::new();
        let core_proof = prover.prove_core(&pk, &stdin, opts, context).unwrap();
        // let public_values = core_proof.public_values.clone();

        tracing::info!("verify core");
        prover.verify(&core_proof.proof, &vk).unwrap();
    }
}
