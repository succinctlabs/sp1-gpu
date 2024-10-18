use moongate_core::cuda_runtime::stream::CudaStream;
use moongate_core::device::memory::ToDevice;
use moongate_core::matrix::ColMajorMatrixDevice;
use moongate_core::stark::trace::AccelAir;
use once_cell::sync::Lazy;
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;
use rayon::prelude::*;
use sp1_core_executor::programs::tests::fibonacci_program;
use sp1_core_executor::ExecutionRecord;
use sp1_core_executor::Executor;
use sp1_core_machine::riscv::RiscvAir;
use sp1_stark::air::MachineAir;
use sp1_stark::baby_bear_poseidon2::BabyBearPoseidon2;
use sp1_stark::SP1CoreOpts;
use sp1_stark::StarkMachine;

struct Env {
    record: ExecutionRecord,
    machine: StarkMachine<BabyBearPoseidon2, RiscvAir<BabyBear>>,
}

static SHARD: Lazy<Env> = Lazy::new(|| {
    let mut executor = Executor::new(fibonacci_program(), SP1CoreOpts::default());
    executor.run().unwrap();
    let record = executor.record;

    let config = BabyBearPoseidon2::new();
    let machine = RiscvAir::machine(config);

    Env { record, machine }
});

#[divan::bench]
fn host(bencher: divan::Bencher) {
    let env = Lazy::force(&SHARD);

    let work = || host_work(env);

    // Warm up.
    for _ in 0..5 {
        work();
    }
    bencher.bench(work);
}

fn host_work(env: &Env) -> Vec<ColMajorMatrixDevice<BabyBear>> {
    env.machine
        .chips()
        .par_iter()
        .map(|chip| {
            let trace: RowMajorMatrix<BabyBear> =
                chip.generate_trace(&env.record, &mut ExecutionRecord::default());
            let mat = trace.to_device().unwrap().to_column_major();
            mat.stream().synchronize().unwrap();
            mat
        })
        .collect()
}

#[divan::bench]
fn on_device(bencher: divan::Bencher) {
    let env = Lazy::force(&SHARD);

    let work = || on_device_work(env);

    // Warm up.
    for _ in 0..5 {
        work();
    }
    bencher.bench(work);
}

fn on_device_work(env: &Env) -> Vec<ColMajorMatrixDevice<BabyBear>> {
    env.machine
        .chips()
        .par_iter()
        .map(|chip| {
            let mat = chip
                .inner()
                .generate_trace_accel(
                    &env.record,
                    &mut ExecutionRecord::default(),
                    &CudaStream::default(),
                )
                .unwrap();
            mat.stream().synchronize().unwrap();
            mat
        })
        .collect()
}
