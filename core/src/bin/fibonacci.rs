use std::env;

use sp1_core::{
    runtime::Program,
    stark::{MachineRecord, RiscvAir, StarkGenericConfig},
    utils::{tests::FIBONACCI_ELF, BabyBearPoseidon2, SP1ProverOpts},
};
use tracing::info;

use moongate_core::{
    stark::StarkGpuProver,
    utils::{execute_core, init_tracer},
};

type SC = BabyBearPoseidon2;

fn main() {
    let program = Program::from(FIBONACCI_ELF);

    env::set_var("RUST_LOG", "debug");

    let config = SC::default();
    let machine = RiscvAir::machine(config);
    let prover = StarkGpuProver::new(machine);

    let (pk, vk) = prover.setup(&program);
    // Execute the program.
    let record = execute_core(program);

    init_tracer();

    let stats = record.stats();
    let cycles = stats.get("cpu_events").unwrap();

    let mut challenger = prover.config().challenger();

    let e2e_time = std::time::Instant::now();
    let proof = prover.prove(
        &pk,
        record,
        &mut challenger,
        SP1ProverOpts::default().core_opts,
    );
    let e2e = e2e_time.elapsed();

    let mut challenger = prover.config().challenger();
    prover
        .machine()
        .verify(&vk, &proof, &mut challenger)
        .unwrap();

    info!(
        "Summary: cycles={}, e2e={:?}, khz={:.2}",
        cycles,
        e2e,
        (*cycles as f64 / (e2e.as_millis() as f64)),
    )
}
