use report::Measurement;
use sp1_core::{runtime::SP1Context, utils::SP1ProverOpts};
use sp1_prover::{components::SP1ProverComponents, SP1Prover, SP1Stdin};

pub mod logger;
pub mod programs;
pub mod report;

pub fn make_measurement<C: SP1ProverComponents>(
    prover: &SP1Prover<C>,
    name: &str,
    elf: &[u8],
    opts: SP1ProverOpts,
) -> Measurement {
    tracing::info!("Starting measurement for {}", name);

    let context = SP1Context::default();

    tracing::info!("Setup elf");
    let (pk, vk) = prover.setup(elf);

    tracing::info!("prove core");
    let time = std::time::Instant::now();
    let stdin = SP1Stdin::new();
    let core_proof = prover.prove_core(&pk, &stdin, opts, context).unwrap();
    let core_time = time.elapsed();

    let cycles = core_proof.cycles as usize;
    let num_shards = core_proof.proof.0.len();

    tracing::info!("verify core");
    prover.verify(&core_proof.proof, &vk).unwrap();

    tracing::info!("compress");
    let time = std::time::Instant::now();
    let compressed_proof = prover.compress(&vk, core_proof, vec![], opts).unwrap();
    let compress_time = time.elapsed();

    tracing::info!("verify compressed");
    prover.verify_compressed(&compressed_proof, &vk).unwrap();

    Measurement {
        name: name.to_string(),
        num_shards,
        cycles,
        core_time,
        compress_time,
    }
}
