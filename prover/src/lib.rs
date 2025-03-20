use components::GpuProverComponents;
use moongate_core::device::memory::cuda_mem_get_info;
use sp1_prover::SP1Prover;
use sp1_stark::SP1ProverOpts;

pub mod components;

pub type SP1GpuProver = SP1Prover<GpuProverComponents>;

/// Get the optimal options for the GPU prover automatically.
pub fn gpu_prover_opts() -> SP1ProverOpts {
    // Convert bytes to GB.
    let gb = 1024.0 * 1024.0 * 1024.0;

    // Get the amount of memory on CPU.
    let cpu_memory_gb: usize =
        ((sysinfo::System::new_all().total_memory() as f64) / gb).ceil() as usize;

    // Get the amount of memory on the GPU.
    let gpu_memory_gb: usize = (((cuda_mem_get_info().unwrap().1 as f64) / gb).ceil() as usize) + 4;

    // Log the memory on CPU and GPU.
    tracing::info!("cpu_memory_gb={}, gpu_memory_gb={}", cpu_memory_gb, gpu_memory_gb);

    SP1ProverOpts::gpu(cpu_memory_gb, gpu_memory_gb)
}

#[cfg(test)]
mod tests {
    use std::env;

    use crate::{components::GpuProverComponents, gpu_prover_opts};
    // use moongate_core::utils::init_tracer;

    use p3_field::PrimeField32;
    use serial_test::serial;
    use sp1_core_executor::RiscvAirId;
    use sp1_core_machine::riscv::RiscvAir;
    use sp1_core_machine::shape::create_dummy_program;
    use sp1_core_machine::shape::create_dummy_record;
    use sp1_prover::{
        // tests::{bench_e2e_prover, test_e2e_prover, test_e2e_with_deferred_proofs_prover, Test},
        SP1Prover,
    };
    use sp1_stark::shape::Shape;
    use sp1_stark::{Dom, MachineProver, StarkGenericConfig};

    // const FIBONACCI_ELF: &[u8] =
    //     include_bytes!("../../perf/programs/fibonacci/riscv32im-succinct-zkvm-elf");

    // const TENDERMINT_BENCHMARK_ELF: &[u8] =
    //     include_bytes!("../../perf/programs/tendermint-benchmark/riscv32im-succinct-zkvm-elf");

    // const RETH_ELF: &[u8] = include_bytes!("../../perf/programs/reth/riscv32im-succinct-zkvm-elf");

    // const KEYSPACE_RECORD_ELF: &[u8] =
    //     include_bytes!("../../perf/programs/keyspace-record/riscv32im-succinct-zkvm-elf");

    // const KEYSPACE_RECORD_INPUT: &[u8] =
    //     include_bytes!("../../perf/programs/keyspace-record/stdin.bin");

    // const KEYSPACE_BATCH_ELF: &[u8] =
    //     include_bytes!("../../perf/programs/keyspace-batcher/riscv32im-succinct-zkvm-elf");

    // const KEYSPACE_BATCH_INPUT: &[u8] =
    //     include_bytes!("../../perf/programs/keyspace-batcher/stdin.bin");

    #[test]
    #[serial]
    fn test_gpu_prover_opts() {
        let opts = gpu_prover_opts();
        println!("{:?}", opts);
    }

    // #[test]
    // #[serial]
    // fn test_e2e_fibonacci() {
    //     let elf = FIBONACCI_ELF;
    //     init_tracer();

    //     let opts = gpu_prover_opts();
    //     let stdin = SP1Stdin::new();
    //     let prover = SP1Prover::<GpuProverComponents>::new();
    //     test_e2e_prover::<GpuProverComponents>(&prover, elf, stdin, opts, Test::Wrap).unwrap()
    // }

    // #[test]
    // #[serial]
    // fn test_e2e_tendermint() {
    //     let elf = TENDERMINT_BENCHMARK_ELF;
    //     init_tracer();

    //     let opts = gpu_prover_opts();
    //     let stdin = SP1Stdin::new();
    //     let prover = SP1Prover::<GpuProverComponents>::new();
    //     test_e2e_prover::<GpuProverComponents>(&prover, elf, stdin, opts, Test::Wrap).unwrap()
    // }

    // #[test]
    // #[serial]
    // #[ignore]
    // fn test_e2e_reth() {
    //     let elf = RETH_ELF;
    //     init_tracer();

    //     let opts = gpu_prover_opts();
    //     let stdin = SP1Stdin::new();
    //     let prover = SP1Prover::<GpuProverComponents>::new();
    //     test_e2e_prover::<GpuProverComponents>(&prover, elf, stdin, opts, Test::Wrap).unwrap()
    // }

    // #[test]
    // #[serial]
    // #[ignore]
    // fn test_e2e_keyspace_record() {
    //     let elf = KEYSPACE_RECORD_ELF;
    //     init_tracer();

    //     let opts = gpu_prover_opts();
    //     let stdin = bincode::deserialize::<SP1Stdin>(KEYSPACE_RECORD_INPUT).unwrap();
    //     let prover = SP1Prover::<GpuProverComponents>::new();
    //     test_e2e_prover::<GpuProverComponents>(&prover, elf, stdin.clone(), opts, Test::Wrap)
    //         .unwrap();
    // }

    // #[test]
    // #[ignore]
    // fn test_e2e_keyspace_batcher() {
    //     let elf = KEYSPACE_BATCH_ELF;
    //     init_tracer();

    //     let opts = gpu_prover_opts();
    //     let stdin = bincode::deserialize::<SP1Stdin>(KEYSPACE_BATCH_INPUT).unwrap();
    //     let prover = SP1Prover::<GpuProverComponents>::new();
    //     test_e2e_prover::<GpuProverComponents>(&prover, elf, stdin.clone(), opts, Test::Wrap)
    //         .unwrap();
    // }

    // #[test]
    // #[serial]
    // fn test_deferred_e2e() {
    //     init_tracer();
    //     let opts = gpu_prover_opts();
    //     test_e2e_with_deferred_proofs_prover::<GpuProverComponents>(opts).unwrap()
    // }

    // fn test_core_elf(elf: &[u8]) {
    //     init_tracer();
    //     let opts = gpu_prover_opts();
    //     let prover = SP1Prover::<GpuProverComponents>::new();
    //     test_e2e_prover::<GpuProverComponents>(&prover, elf, SP1Stdin::new(), opts, Test::Core)
    //         .unwrap()
    // }

    // fn test_compress_elf(elf: &[u8]) {
    //     init_tracer();
    //     let opts = gpu_prover_opts();
    //     let prover = SP1Prover::<GpuProverComponents>::new();
    //     test_e2e_prover::<GpuProverComponents>(&prover, elf, SP1Stdin::new(), opts, Test::Compress)
    //         .unwrap()
    // }

    // fn bench_elf(elf: &[u8], kind: Test) {
    //     init_tracer();
    //     let opts = gpu_prover_opts();
    //     let prover = SP1Prover::<GpuProverComponents>::new();
    //     bench_e2e_prover::<GpuProverComponents>(&prover, elf, SP1Stdin::new(), opts, kind).unwrap()
    // }

    // #[test]
    // #[serial]
    // fn test_core_fibonacci() {
    //     test_core_elf(FIBONACCI_ELF);
    // }

    // #[test]
    // #[serial]
    // #[ignore]
    // fn bench_core_fibonacci() {
    //     bench_elf(FIBONACCI_ELF, Test::Core);
    // }

    // #[test]
    // #[serial]
    // #[ignore]
    // fn test_compress_tendermint() {
    //     test_compress_elf(TENDERMINT_BENCHMARK_ELF);
    // }

    // #[test]
    // #[serial]
    // #[ignore]
    // fn bench_compress_tendermint() {
    //     bench_elf(TENDERMINT_BENCHMARK_ELF, Test::Compress);
    // }

    // #[test]
    // #[serial]
    // #[ignore]
    // fn bench_core_reth() {
    //     bench_elf(RETH_ELF, Test::Core);
    // }

    // #[test]
    // #[serial]
    // #[ignore]
    // fn bench_compress_reth() {
    //     bench_elf(RETH_ELF, Test::Compress);
    // }

    fn try_generate_dummy_proof<SC: StarkGenericConfig, P: MachineProver<SC, RiscvAir<SC::Val>>>(
        prover: &P,
        shape: &Shape<RiscvAirId>,
    ) where
        SC::Val: PrimeField32,
        Dom<SC>: std::fmt::Debug,
    {
        let program = create_dummy_program(shape);
        let mut record = create_dummy_record(shape);

        // Try doing setup.
        let (pk, _) = prover.setup(&program);

        // Try to generate traces.
        let main_traces = prover.generate_traces(&mut record);

        // Try to commit the traces.
        let main_data = prover.commit(&record, main_traces);

        let mut challenger = prover.machine().config().challenger();

        // Try to "open".
        prover.open(&pk, main_data, &mut challenger).unwrap();
    }

    #[test]
    #[ignore]
    fn test_shapes() {
        env::set_var("FIX_CORE_SHAPES", "true");
        env::set_var("FIX_RECURSION_SHAPES", "true");
        println!("hello world");
        let prover = SP1Prover::<GpuProverComponents>::new();

        let shape_config = prover.core_shape_config.as_ref().unwrap();

        let mut pks = vec![];
        for program in prover.join_programs_map.values() {
            let (pk, _) = prover.compress_prover.setup(program);
            pks.push(pk);
        }
        for (i, program) in prover.join_programs_map.values().enumerate() {
            if i >= 2 {
                continue;
            }
            let (pk, _) = prover.compress_prover.setup(program);
            pks.push(pk);
        }
        println!("pks={:?}", pks.len());

        // let blacklist = [33, 144, 155, 158, 195];
        for (i, shape) in shape_config.maximal_core_shapes(21).into_iter().enumerate() {
            // if i <= *blacklist.last().unwrap() {
            //     continue;
            // }
            println!("finished shape: id={}, shape={:?}", i, shape);
            try_generate_dummy_proof(&prover.core_prover, &shape);
        }
    }
}
