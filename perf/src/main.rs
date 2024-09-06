use moongate_core::utils::init_tracer;
use moongate_perf::programs::{KEYSPACE_BATCHER_ELF, KEYSPACE_ELF};
use sp1_prover::SP1Prover;

use moongate_prover::{components::GpuProverComponents, gpu_prover_opts};

use clap::{Parser, ValueEnum};
use moongate_perf::report::write_measurements_to_csv;
use moongate_perf::tracer;
use moongate_perf::{
    make_measurement,
    programs::{FIBONACCI_ELF, LOOP_ELF, RETH_ELF, SHA2_CHAIN_ELF, TENDERMINT_BENCHMARK_ELF},
};
use opentelemetry::KeyValue;
use opentelemetry_sdk::Resource;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "all")]
    pub program: Program,
    #[arg(short, long, default_value = "telemetry")]
    pub trace: Trace,
}

#[derive(Clone, Debug, ValueEnum)]
enum Program {
    Fibonacci,
    Loop,
    Sha2Chain,
    Tendermint,
    Reth,
    KeyspaceRecord,
    KeyspaceBatcher,
    All,
}

#[derive(Clone, Debug, ValueEnum)]
enum Trace {
    Nvtx,
    Telemetry,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize the tracer.
    match args.trace {
        Trace::Nvtx => init_tracer(),
        Trace::Telemetry => {
            let resource = Resource::new(vec![KeyValue::new("service.name", "moongate-perf")]);
            tracer::init(resource);
        }
    }

    let named_programs = match args.program {
        Program::All => vec![
            ("Fibonacci", FIBONACCI_ELF),
            ("Loop", LOOP_ELF),
            ("Sha2 Chain", SHA2_CHAIN_ELF),
            ("Tendermint Benchmark", TENDERMINT_BENCHMARK_ELF),
            ("Reth", RETH_ELF),
        ],
        Program::Fibonacci => vec![("Fibonacci", FIBONACCI_ELF)],
        Program::Loop => vec![("Loop", LOOP_ELF)],
        Program::Sha2Chain => vec![("Sha2 Chain", SHA2_CHAIN_ELF)],
        Program::Tendermint => {
            vec![("Tendermint Benchmark", TENDERMINT_BENCHMARK_ELF)]
        }
        Program::Reth => vec![("Reth", RETH_ELF)],
        Program::KeyspaceRecord => vec![("KeyspaceRecord", KEYSPACE_ELF)],
        Program::KeyspaceBatcher => vec![("KeyspaceBatcher", KEYSPACE_BATCHER_ELF)],
    };

    let prover: SP1Prover<GpuProverComponents> = SP1Prover::new();
    let opts = gpu_prover_opts();

    let mut measurements = vec![];
    for (name, elf) in named_programs {
        let measurement = make_measurement(&prover, name, elf, opts);
        println!("{}", measurement);
        measurements.push(measurement);
        // Make a second measurement to see affects of the cache.
        // let measurement = make_measurement(&prover, name, elf, opts);
        // println!("{}", measurement);
        // measurements.push(measurement);
    }

    write_measurements_to_csv(&measurements, "measurements.csv")?;

    Ok(())
}
