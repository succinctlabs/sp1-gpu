use std::sync::Arc;

use report::Measurement;
use sp1_core::{
    runtime::{Program, SP1Context},
    utils::SP1ProverOpts,
};
use sp1_prover::{
    components::SP1ProverComponents, CoreSC, SP1CoreProof, SP1CoreProofData, SP1Prover,
    SP1PublicValues, SP1Stdin,
};

pub mod programs;
pub mod report;

pub fn make_measurement<C: SP1ProverComponents>(
    prover: &SP1Prover<C>,
    name: &str,
    elf: &[u8],
    opts: SP1ProverOpts,
) -> Measurement {
    tracing::info!("Starting measurement for {}", name);

    let mut context = SP1Context::default();

    tracing::info!("Setup elf");
    let (pk, vk) = prover.setup(elf);

    tracing::info!("prove core");
    let time = std::time::Instant::now();
    let stdin = SP1Stdin::new();
    // let core_proof = prover.prove_core(&pk, &stdin, opts, context).unwrap();
    context
        .subproof_verifier
        .get_or_insert_with(|| Arc::new(prover));
    let config = CoreSC::default();
    let program = Program::from(&pk.elf);
    let (proof, public_values_stream, cycles) = sp1_core::utils::prove_with_context::<
        _,
        C::CoreProver,
    >(
        program, &stdin, config, opts.core_opts, context
    )
    .unwrap();
    let public_values = SP1PublicValues::from(&public_values_stream);
    let core_proof = SP1CoreProof {
        proof: SP1CoreProofData(proof.shard_proofs),
        stdin: stdin.clone(),
        public_values,
    };
    let core_time = time.elapsed();

    let num_shards = core_proof.proof.0.len();

    tracing::info!("verify core");
    prover.verify(&core_proof.proof, &vk).unwrap();

    tracing::info!("compress");
    let time = std::time::Instant::now();
    let compressed_proof = prover.compress(&vk, core_proof, vec![], opts).unwrap();
    let compress_time = time.elapsed();

    tracing::info!("verify compressed");
    prover.verify_compressed(&compressed_proof, &vk).unwrap();

    Measurement::new(name, cycles as usize, num_shards, core_time, compress_time)
}
