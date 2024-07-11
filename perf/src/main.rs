use sp1_prover::SP1Prover;

use moongate_core::utils::init_tracer;
use moongate_prover::{components::GpuProverComponents, gpu_prover_opts};

use moongate_perf::report::write_measurements_to_csv;
use moongate_perf::{
    make_measurement,
    programs::{FIBONACCI_ELF, LOOP_ELF, RETH_ELF, SHA2_CHAIN_ELF, TENDERMINT_BENCHMARK_ELF},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracer();
    let mut measurements = vec![];

    let named_programs = [
        ("Fibonacci", FIBONACCI_ELF),
        ("Loop", LOOP_ELF),
        ("Sha2 Chain", SHA2_CHAIN_ELF),
        ("Tendermint Benchmark", TENDERMINT_BENCHMARK_ELF),
        ("Reth", RETH_ELF),
    ];

    tracing::info!("initializing prover");
    let prover: SP1Prover<GpuProverComponents> =
        tracing::info_span!("Initializing prover").in_scope(SP1Prover::new);

    let opts = gpu_prover_opts();

    for (name, elf) in named_programs {
        let measurement = make_measurement(&prover, name, elf, opts);
        println!("{}", measurement);
        measurements.push(measurement);
    }

    write_measurements_to_csv(&measurements, "measurements.csv")?;

    Ok(())
}
