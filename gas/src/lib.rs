use std::time::Duration;

use clap::ValueEnum;
use record::Measurement;
use sp1_core_executor::SP1Context;
use sp1_core_machine::io::SP1Stdin;
use sp1_prover::{components::SP1ProverComponents, SP1Prover};
use sp1_stark::SP1ProverOpts;

pub mod record;
pub mod report;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum Stage {
    Core,
    Compress,
    Shrink,
    Wrap,
}

pub fn make_measurement<C: SP1ProverComponents>(
    prover: &SP1Prover<C>,
    name: &str,
    elf: &[u8],
    stdin: Option<SP1Stdin>,
    opts: SP1ProverOpts,
    stage: Stage,
) -> Measurement {
    tracing::info!("Starting measurement for {}", name);

    let context = SP1Context::default();

    tracing::info!("Setup elf");
    let (_pk_host, pk_device, program, vk) = prover.setup(elf);

    let stdin = stdin.unwrap_or_default();

    tracing::info!("execute");
    prover.execute(elf, &stdin, SP1Context::default()).unwrap();

    tracing::info!("prove core");
    let time = std::time::Instant::now();
    let core_proof = prover.prove_core(&pk_device, program, &stdin, opts, context).unwrap();
    let core_time = time.elapsed();

    let cycles = core_proof.cycles as usize;
    let num_shards = core_proof.proof.0.len();

    if stage == Stage::Core {
        return Measurement {
            name: name.to_string(),
            cycles,
            num_shards,
            core_time,
            core_proving_time: Duration::ZERO,
            compress_time: Duration::ZERO,
            shrink_time: Duration::ZERO,
            wrap_time: Duration::ZERO,
        };
    }

    let deferred_proofs = stdin.proofs.iter().map(|(proof, _)| proof.clone()).collect::<Vec<_>>();
    tracing::info!("compress");

    let core_proof_bytes = bincode::serialize(&core_proof).unwrap();
    std::fs::write("core_proof.bin", &core_proof_bytes).unwrap();

    let time = std::time::Instant::now();
    let compressed_proof = prover.compress(&vk, core_proof, deferred_proofs, opts).unwrap();
    let compress_time = time.elapsed();

    if stage == Stage::Compress {
        return Measurement {
            name: name.to_string(),
            cycles,
            num_shards,
            core_time,
            core_proving_time: Duration::ZERO,
            compress_time,
            shrink_time: Duration::ZERO,
            wrap_time: Duration::ZERO,
        };
    }

    tracing::info!("shrink");
    let time = std::time::Instant::now();
    let shrink_proof = prover.shrink(compressed_proof, opts).unwrap();
    let shrink_time = time.elapsed();

    if stage == Stage::Shrink {
        return Measurement {
            name: name.to_string(),
            cycles,
            num_shards,
            core_time,
            core_proving_time: Duration::ZERO,
            compress_time,
            shrink_time,
            wrap_time: Duration::ZERO,
        };
    }

    tracing::info!("wrap");
    let time = std::time::Instant::now();
    let _ = prover.wrap_bn254(shrink_proof, opts).unwrap();
    let wrap_time = time.elapsed();

    Measurement {
        name: name.to_string(),
        cycles,
        num_shards,
        core_time,
        core_proving_time: Duration::ZERO,
        compress_time,
        shrink_time,
        wrap_time,
    }
}
