use moongate_prover::{components::GpuProverComponents, gpu_prover_opts};
use sp1_core_executor::SP1Context;
use sp1_core_machine::io::SP1Stdin;
use sp1_prover::SP1Prover;

const FIBONACCI_ELF: &[u8] =
    include_bytes!("../../perf/programs/fibonacci/riscv32im-succinct-zkvm-elf");

fn main() {
    divan::main();
}

#[divan::bench]
fn core(bencher: divan::Bencher) {
    let opts = gpu_prover_opts();

    let stdin = SP1Stdin::default();
    let prover: SP1Prover<GpuProverComponents> = SP1Prover::new();

    let (pk, _) = prover.setup(FIBONACCI_ELF);

    let work =
        || divan::black_box(prover.prove_core(&pk, &stdin, opts, SP1Context::default()).unwrap());
    // Warm up.
    for _ in 0..5 {
        work();
    }
    bencher.bench(work);
}
