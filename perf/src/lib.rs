use std::time::Duration;

use programs::{KEYSPACE_BATCHER_STDIN, KEYSPACE_STDIN};
use report::Measurement;
use sp1_core_executor::SP1Context;
use sp1_core_machine::io::SP1Stdin;
use sp1_prover::{components::SP1ProverComponents, SP1Prover};
use sp1_stark::SP1ProverOpts;

pub mod programs;
pub mod report;
pub mod tracer;

pub fn make_measurement<C: SP1ProverComponents>(
    prover: &SP1Prover<C>,
    name: &str,
    elf: &[u8],
    stdin: Option<SP1Stdin>,
    opts: SP1ProverOpts,
    verify: bool,
) -> Measurement {
    tracing::info!("Starting measurement for {}", name);

    let context = SP1Context::default();

    tracing::info!("Setup elf");
    let (pk, vk) = prover.setup(elf);

    tracing::info!("prove core");
    let time = std::time::Instant::now();
    let mut stdin = stdin.unwrap_or_default();
    if name == "KeyspaceRecord" {
        stdin = bincode::deserialize(KEYSPACE_STDIN).unwrap();
    } else if name == "KeyspaceBatcher" {
        stdin = bincode::deserialize(KEYSPACE_BATCHER_STDIN).unwrap();
    }
    let core_proof = prover.prove_core(&pk, &stdin, opts, context).unwrap();
    let core_time = time.elapsed();

    let cycles = core_proof.cycles as usize;
    let num_shards = core_proof.proof.0.len();

    if verify {
        tracing::info!("verify core");
        prover.verify(&core_proof.proof, &vk).unwrap();
    }

    tracing::info!("compress");
    let time = std::time::Instant::now();
    let compressed_proof = prover.compress(&vk, core_proof, vec![], opts).unwrap();
    let compress_time = time.elapsed();

    if verify {
        tracing::info!("verify compressed");
        prover.verify_compressed(&compressed_proof, &vk).unwrap();
    }

    Measurement { name: name.to_string(), num_shards, cycles, core_time, compress_time }
}
