use std::env;

use components::GpuProverComponents;
use moongate_core::device::memory::cuda_mem_get_info;
use sp1_prover::SP1Prover;
use sp1_stark::{SP1ProverOpts, SplitOpts};

pub mod components;

pub type SP1GpuProver = SP1Prover<GpuProverComponents>;

const SHARD_MEM_CONSTANT: usize = 20;

const SHARD_MEM_RATIO: f64 = (1 << 20) as f64 / (23.0 * 1e9);
const DEFFERRED_SPLIT_LOG_RATIO: usize = 4;
const MAX_SHARD_SIZE: usize = 1 << 22;

pub fn gpu_prover_opts() -> SP1ProverOpts {
    let mut opts = SP1ProverOpts::default();

    // Core options
    let (_, total) = cuda_mem_get_info().unwrap();
    tracing::info!("Total memory on device: {}", total);

    let shard_size_log = ((total as f64) * SHARD_MEM_RATIO).log2().floor() as usize;
    let default_shard_size = 1 << shard_size_log;
    let shard_size = env::var("SHARD_SIZE")
        .map_or_else(|_| default_shard_size, |s| s.parse::<usize>().unwrap_or(default_shard_size));
    let shard_size = std::cmp::min(shard_size, MAX_SHARD_SIZE);
    opts.core_opts.shard_size = shard_size;
    tracing::info!("Shard size set to {}", shard_size);
    opts.core_opts.shard_batch_size = 1;

    // Set the deferred split threshold.
    let deferred_split_threshold_log = shard_size_log - DEFFERRED_SPLIT_LOG_RATIO;
    let default_deferred_split_threshold = 1 << deferred_split_threshold_log;
    let deferred_split_threshold = env::var("SPLIT_THRESHOLD")
        .map(|s| s.parse::<usize>().unwrap_or(default_deferred_split_threshold))
        .unwrap_or(default_deferred_split_threshold);
    tracing::info!("Deffered split threshold set to {}", deferred_split_threshold);
    opts.core_opts.split_opts = SplitOpts::new(deferred_split_threshold);

    opts.core_opts.records_and_traces_channel_capacity = 4;
    opts.core_opts.trace_gen_workers = 4;

    opts.recursion_opts.shard_batch_size = 1;

    let s = sysinfo::System::new_all();
    let total_memory_gb = (s.total_memory() as f64) / (1024.0 * 1024.0 * 1024.0);
    if total_memory_gb < 20.0 {
        opts.recursion_opts.records_and_traces_channel_capacity = 1;
        opts.recursion_opts.trace_gen_workers = 1;
    } else {
        opts.recursion_opts.records_and_traces_channel_capacity = 4;
        opts.recursion_opts.trace_gen_workers = 4;
    }
    opts
}

#[cfg(test)]
mod tests {
    use crate::components::GpuProverComponents;
    use crate::gpu_prover_opts;
    use moongate_core::utils::init_tracer;
    use p3_baby_bear::BabyBear;
    use sp1_core_executor::CoreShape;
    use sp1_core_machine::alu::{
        AddSubChip, BitwiseChip, DivRemChip, LtChip, MulChip, ShiftLeft, ShiftRightChip,
    };
    use sp1_core_machine::cpu::CpuChip;
    use sp1_core_machine::io::SP1Stdin;
    use sp1_core_machine::memory::{MemoryLocalChip, MemoryProgramChip};
    use sp1_core_machine::riscv::tests::try_generate_dummy_proof;
    use sp1_core_machine::riscv::{ByteChip, ProgramChip, RiscvAir, SyscallChip};
    use sp1_core_machine::utils::tests::FIBONACCI_ELF;
    use sp1_prover::tests::{
        bench_e2e_prover, test_e2e_prover, test_e2e_with_deferred_proofs_prover, Test,
    };
    use sp1_prover::SP1Prover;
    use sp1_stark::air::MachineAir;

    const TENDERMINT_BENCHMARK_ELF: &[u8] =
        include_bytes!("../../perf/programs/tendermint-benchmark/riscv32im-succinct-zkvm-elf");

    const RETH_ELF: &[u8] = include_bytes!("../../perf/programs/reth/riscv32im-succinct-zkvm-elf");

    const KEYSPACE_RECORD_ELF: &[u8] =
        include_bytes!("../../perf/programs/keyspace-record/riscv32im-succinct-zkvm-elf");

    const KEYSPACE_RECORD_INPUT: &[u8] =
        include_bytes!("../../perf/programs/keyspace-record/stdin.bin");

    const KEYSPACE_BATCH_ELF: &[u8] =
        include_bytes!("../../perf/programs/keyspace-batcher/riscv32im-succinct-zkvm-elf");

    const KEYSPACE_BATCH_INPUT: &[u8] =
        include_bytes!("../../perf/programs/keyspace-batcher/stdin.bin");

    #[test]
    fn test_gpu_prover_opts() {
        let opts = gpu_prover_opts();
        println!("{:?}", opts);
    }

    #[test]
    fn test_e2e_fibonacci() {
        let elf = FIBONACCI_ELF;
        init_tracer();

        let opts = gpu_prover_opts();
        let stdin = SP1Stdin::new();
        let prover = SP1Prover::<GpuProverComponents>::new();
        test_e2e_prover::<GpuProverComponents>(&prover, elf, stdin, opts, Test::Wrap).unwrap()
    }

    #[test]
    fn test_e2e_tendermint() {
        let elf = TENDERMINT_BENCHMARK_ELF;
        init_tracer();

        let opts = gpu_prover_opts();
        let stdin = SP1Stdin::new();
        let prover = SP1Prover::<GpuProverComponents>::new();
        test_e2e_prover::<GpuProverComponents>(&prover, elf, stdin, opts, Test::Wrap).unwrap()
    }

    #[test]
    #[ignore]
    fn test_e2e_reth() {
        let elf = RETH_ELF;
        init_tracer();

        let opts = gpu_prover_opts();
        let stdin = SP1Stdin::new();
        let prover = SP1Prover::<GpuProverComponents>::new();
        test_e2e_prover::<GpuProverComponents>(&prover, elf, stdin, opts, Test::Wrap).unwrap()
    }

    #[test]
    #[ignore]
    fn test_e2e_keyspace_record() {
        let elf = KEYSPACE_RECORD_ELF;
        init_tracer();

        let opts = gpu_prover_opts();
        let stdin = bincode::deserialize::<SP1Stdin>(KEYSPACE_RECORD_INPUT).unwrap();
        let prover = SP1Prover::<GpuProverComponents>::new();
        test_e2e_prover::<GpuProverComponents>(&prover, elf, stdin.clone(), opts, Test::Wrap)
            .unwrap();
    }

    #[test]
    #[ignore]
    fn test_e2e_keyspace_batcher() {
        let elf = KEYSPACE_BATCH_ELF;
        init_tracer();

        let opts = gpu_prover_opts();
        let stdin = bincode::deserialize::<SP1Stdin>(KEYSPACE_BATCH_INPUT).unwrap();
        let prover = SP1Prover::<GpuProverComponents>::new();
        test_e2e_prover::<GpuProverComponents>(&prover, elf, stdin.clone(), opts, Test::Wrap)
            .unwrap();
    }

    #[test]
    fn test_deferred_e2e() {
        init_tracer();
        let opts = gpu_prover_opts();
        test_e2e_with_deferred_proofs_prover::<GpuProverComponents>(opts).unwrap()
    }

    fn test_core_elf(elf: &[u8]) {
        init_tracer();
        let opts = gpu_prover_opts();
        let prover = SP1Prover::<GpuProverComponents>::new();
        test_e2e_prover::<GpuProverComponents>(&prover, elf, SP1Stdin::new(), opts, Test::Core)
            .unwrap()
    }

    fn test_compress_elf(elf: &[u8]) {
        init_tracer();
        let opts = gpu_prover_opts();
        let prover = SP1Prover::<GpuProverComponents>::new();
        test_e2e_prover::<GpuProverComponents>(&prover, elf, SP1Stdin::new(), opts, Test::Compress)
            .unwrap()
    }

    fn bench_elf(elf: &[u8], kind: Test) {
        init_tracer();
        let opts = gpu_prover_opts();
        let prover = SP1Prover::<GpuProverComponents>::new();
        bench_e2e_prover::<GpuProverComponents>(&prover, elf, SP1Stdin::new(), opts, kind).unwrap()
    }

    #[test]
    fn test_core_fibonacci() {
        test_core_elf(FIBONACCI_ELF);
    }

    #[test]
    #[ignore]
    fn bench_core_fibonacci() {
        bench_elf(FIBONACCI_ELF, Test::Core);
    }

    #[test]
    #[ignore]
    fn test_compress_tendermint() {
        test_compress_elf(TENDERMINT_BENCHMARK_ELF);
    }

    #[test]
    #[ignore]
    fn bench_compress_tendermint() {
        bench_elf(TENDERMINT_BENCHMARK_ELF, Test::Compress);
    }

    #[test]
    #[ignore]
    fn bench_core_reth() {
        bench_elf(RETH_ELF, Test::Core);
    }

    #[test]
    #[ignore]
    fn bench_compress_reth() {
        bench_elf(RETH_ELF, Test::Compress);
    }

    #[test]
    #[ignore]
    #[allow(clippy::default_constructed_unit_structs)]
    fn test_shapes() {
        init_tracer();
        let prover = SP1Prover::<GpuProverComponents>::new();

        let fixed_log_heights = [
            (RiscvAir::<BabyBear>::Program(ProgramChip::default()), 22),
            (RiscvAir::<BabyBear>::ProgramMemory(MemoryProgramChip::default()), 22),
            (RiscvAir::<BabyBear>::ByteLookup(ByteChip::default()), 16),
            (RiscvAir::<BabyBear>::SyscallCore(SyscallChip::core()), 20),
            (RiscvAir::<BabyBear>::Cpu(CpuChip::default()), 21),
            (RiscvAir::<BabyBear>::MemoryLocal(MemoryLocalChip::new()), 21),
        ];

        let variable_log_heights = [
            (RiscvAir::<BabyBear>::Add(AddSubChip::default()), 20),
            (RiscvAir::<BabyBear>::DivRem(DivRemChip::default()), 20),
            (RiscvAir::<BabyBear>::Bitwise(BitwiseChip::default()), 21),
            (RiscvAir::<BabyBear>::Mul(MulChip::default()), 20),
            (RiscvAir::<BabyBear>::ShiftRight(ShiftRightChip::default()), 20),
            (RiscvAir::<BabyBear>::ShiftLeft(ShiftLeft::default()), 20),
            (RiscvAir::<BabyBear>::Lt(LtChip::default()), 20),
        ];

        let height_map = fixed_log_heights
            .into_iter()
            .chain(variable_log_heights)
            .map(|(air, log_height)| (air.name(), log_height));

        let shape = CoreShape { inner: height_map.collect() };

        try_generate_dummy_proof(&prover.core_prover, &shape);
    }
}
