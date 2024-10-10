use moongate_core::device::memory::ToDevice;
use once_cell::sync::Lazy;
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;
use rand::thread_rng;
use rand::Rng;
use sp1_core_executor::events::AluEvent;
use sp1_core_executor::ExecutionRecord;
use sp1_core_executor::Opcode;
use sp1_core_machine::alu::AddSubChip;
use sp1_stark::air::MachineAir;

fn main() {
    divan::main();
}

const NUM_OPS_EACH: u32 = 16384;

static SHARD: Lazy<ExecutionRecord> = Lazy::new(|| {
    let mut shard = ExecutionRecord::default();
    for i in 0..NUM_OPS_EACH {
        let operand_1 = thread_rng().gen_range(0..u32::MAX);
        let operand_2 = thread_rng().gen_range(0..u32::MAX);
        let result = operand_1.wrapping_add(operand_2);
        shard.add_events.push(AluEvent::new(i % 2, 0, Opcode::ADD, result, operand_1, operand_2));
    }
    for i in 0..NUM_OPS_EACH {
        let operand_1 = thread_rng().gen_range(0..u32::MAX);
        let operand_2 = thread_rng().gen_range(0..u32::MAX);
        let result = operand_1.wrapping_sub(operand_2);
        shard.add_events.push(AluEvent::new(i % 2, 0, Opcode::SUB, result, operand_1, operand_2));
    }
    shard
});

#[divan::bench]
fn baseline(bencher: divan::Bencher) {
    let shard = Lazy::force(&SHARD);

    let work = || {
        let chip = AddSubChip;
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(shard, &mut ExecutionRecord::default());
        trace.to_device().unwrap().to_column_major()
    };

    // Warm up.
    for _ in 0..5 {
        work();
    }
    bencher.bench(work);
}
