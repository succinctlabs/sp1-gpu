use moongate_core::cuda_runtime::stream::CudaStream;
use moongate_core::device::memory::ToDevice;
use moongate_core::matrix::ColMajorMatrixDevice;
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;
use rand::thread_rng;
use rand::Rng;
use sp1_core_executor::events::AluEvent;
use sp1_core_executor::ExecutionRecord;
use sp1_core_executor::Opcode;
use sp1_core_machine::alu::AddSubChip;
use sp1_stark::air::MachineAir;
use std::sync::LazyLock;

const NUM_OPS_EACH: u32 = 100_000;
static SHARD: LazyLock<ExecutionRecord> = LazyLock::new(|| {
    let add_sub_events = (0..NUM_OPS_EACH)
        .flat_map(|i| {
            [
                {
                    let operand_1 = thread_rng().gen_range(0..u32::MAX);
                    let operand_2 = thread_rng().gen_range(0..u32::MAX);
                    let result = operand_1.wrapping_add(operand_2);
                    AluEvent::new(i % 2, 0, Opcode::ADD, result, operand_1, operand_2)
                },
                {
                    let operand_1 = thread_rng().gen_range(0..u32::MAX);
                    let operand_2 = thread_rng().gen_range(0..u32::MAX);
                    let result = operand_1.wrapping_sub(operand_2);
                    AluEvent::new(i % 2, 0, Opcode::SUB, result, operand_1, operand_2)
                },
            ]
        })
        .collect::<Vec<_>>();
    ExecutionRecord { add_sub_events, ..Default::default() }
});

#[divan::bench]
fn host(bencher: divan::Bencher) {
    let shard = LazyLock::force(&SHARD);

    let work = || divan::black_box(host_work(shard));

    // Warm up.
    for _ in 0..5 {
        work();
    }
    bencher.bench(work);
}

fn host_work(shard: &ExecutionRecord) -> ColMajorMatrixDevice<BabyBear> {
    let chip = AddSubChip;
    let trace: RowMajorMatrix<BabyBear> =
        chip.generate_trace(shard, &mut ExecutionRecord::default());
    let mat = trace.to_device().unwrap().to_column_major();
    mat.stream().synchronize().unwrap();
    mat
}

#[divan::bench]
fn on_device(bencher: divan::Bencher) {
    let shard = LazyLock::force(&SHARD);

    let work = || divan::black_box(on_device_work(shard));

    // Warm up.
    for _ in 0..5 {
        work();
    }
    bencher.bench(work);
}

fn on_device_work(shard: &ExecutionRecord) -> ColMajorMatrixDevice<BabyBear> {
    let mat = moongate_core::stark::trace::add_sub_generate_trace(
        &AddSubChip,
        shard,
        &mut ExecutionRecord::default(),
        &CudaStream::default(),
    )
    .unwrap();
    mat.stream().synchronize().unwrap();
    mat
}
