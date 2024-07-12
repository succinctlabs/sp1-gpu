use moongate_prover::{components::GpuProverComponents, gpu_prover_opts};
use sp1_core::{runtime::SP1Context, utils::tests::FIBONACCI_ELF};
use sp1_prover::{SP1Prover, SP1Stdin};

fn main() {
    tracing::info!("initializing prover");
    let prover: SP1Prover<GpuProverComponents> =
        tracing::info_span!("Initializing prover").in_scope(SP1Prover::new);

    let context = SP1Context::default();

    tracing::info!("Setup elf");
    let (pk, vk) = prover.setup(FIBONACCI_ELF);

    tracing::info!("prove core");
    let stdin = SP1Stdin::new();
    let opts = gpu_prover_opts();
    let core_proof = prover.prove_core(&pk, &stdin, opts, context).unwrap();

    tracing::info!("verify core");
    prover.verify(&core_proof.proof, &vk).unwrap();
}
