use std::env;

use p3_challenger::CanObserve;
use sp1_core::{
    runtime::Program,
    stark::{MachineRecord, RiscvAir, StarkGenericConfig, Verifier},
    utils::{tests::FIBONACCI_ELF, BabyBearPoseidon2},
};
use tracing::{debug_span, info, info_span};

use moongate_core::{
    stark::StarkGpuProver,
    utils::{execute_core, init_tracer},
};

type SC = BabyBearPoseidon2;

fn main() {
    let program = Program::from(FIBONACCI_ELF);

    let config = SC::default();
    let machine = RiscvAir::machine(config);
    let gpu_prover = StarkGpuProver::new(machine);

    let (pk, vk) = gpu_prover.setup(&program);
    // Execute the program.
    let record = execute_core(program);

    env::set_var("RUST_LOG", "debug");

    init_tracer();

    let stats = record.stats();
    let cycles = stats.get("cpu_events").unwrap();

    let e2e_time = std::time::Instant::now();
    let shards = debug_span!("Shard execution trace").in_scope(|| gpu_prover.shard(record));

    let e2e_time_no_shard = std::time::Instant::now();
    for (i, shard) in shards.into_iter().enumerate() {
        let main_data =
            info_span!("Commit_main").in_scope(|| gpu_prover.commit_main(&shard, i + 1));
        // Observe the main commit.
        let main_commit = main_data.commit;
        let mut challenger = gpu_prover.config().challenger();
        challenger.observe(main_commit);
        let proof = info_span!("prove shard").in_scope(|| {
            gpu_prover
                .prove_shard(&pk, main_data, &mut challenger)
                .unwrap()
        });

        // Verify the proof.
        let mut challenger = gpu_prover.config().challenger();
        challenger.observe(main_commit);
        let shard_chips = gpu_prover
            .machine()
            .shard_chips_ordered(&proof.chip_ordering)
            .collect::<Vec<_>>();
        Verifier::<SC, _>::verify_shard(
            gpu_prover.config(),
            &vk,
            &shard_chips,
            &mut challenger,
            &proof,
        )
        .unwrap();
    }
    let e2e = e2e_time.elapsed();
    let e2e_no_shard = e2e_time_no_shard.elapsed();
    info!(
        "Summary: cycles={}, e2e={:?}, khz={:.2}, e2e_no_shard={:?}, khz_no_shard={:.2}",
        cycles,
        e2e,
        (*cycles as f64 / (e2e.as_millis() as f64)),
        e2e_no_shard,
        (*cycles as f64 / (e2e_no_shard.as_millis() as f64)),
    )
}
