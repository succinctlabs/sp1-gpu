use components::GpuProverComponents;
use sp1_core::utils::SP1ProverOpts;
use sp1_prover::SP1Prover;

pub mod components;

pub type SP1GpuProver = SP1Prover<GpuProverComponents>;

pub fn gpu_prover_opts() -> SP1ProverOpts {
    let mut opts = SP1ProverOpts::default();

    opts.core_opts.shard_size = 1 << 21;
    opts.core_opts.shard_batch_size = 4;
    opts.core_opts.split_opts.keccak_split_threshold = (1 << 18) / 24;
    opts.core_opts.stream_capacity = 4;

    opts.recursion_opts.stream_capacity = 8;

    opts
}

#[cfg(test)]
mod tests {
    use moongate_core::utils::init_tracer;
    use sp1_core::utils::tests::FIBONACCI_ELF;
    use sp1_prover::tests::test_e2e_prover;
    use sp1_prover::tests::Test;

    use crate::components::GpuProverComponents;
    use crate::gpu_prover_opts;

    const TENDERMINT_BENCHMARK_ELF: &[u8] =
        include_bytes!("../../../sp1/tests/tendermint-benchmark/elf/riscv32im-succinct-zkvm-elf");

    #[test]
    fn test_e2e_fibonacci() {
        let elf = FIBONACCI_ELF;
        init_tracer();

        let opts = gpu_prover_opts();
        test_e2e_prover::<GpuProverComponents>(elf, opts, Test::Compress).unwrap()
    }

    fn test_core_elf(elf: &[u8]) {
        init_tracer();
        let opts = gpu_prover_opts();
        test_e2e_prover::<GpuProverComponents>(elf, opts, Test::Core).unwrap()
    }

    fn test_compress_elf(elf: &[u8]) {
        init_tracer();
        let opts = gpu_prover_opts();
        test_e2e_prover::<GpuProverComponents>(elf, opts, Test::Compress).unwrap()
    }

    #[test]
    fn test_core_fibonacci() {
        test_core_elf(FIBONACCI_ELF);
    }

    #[test]
    #[ignore]
    fn test_compress_tendermint() {
        test_compress_elf(TENDERMINT_BENCHMARK_ELF);
    }

    #[test]
    #[ignore]
    fn test_core_tendermint() {
        test_core_elf(TENDERMINT_BENCHMARK_ELF);
    }

    #[test]
    #[ignore]
    fn test_core_reth() {
        const RETH_ELF: &[u8] =
            include_bytes!("../../../zkvm-perf/programs/reth-sp1/elf/riscv32im-succinct-zkvm-elf");
        test_core_elf(RETH_ELF);
    }

    #[test]
    #[ignore]
    fn test_compress_reth() {
        const RETH_ELF: &[u8] =
            include_bytes!("../../../zkvm-perf/programs/reth-sp1/elf/riscv32im-succinct-zkvm-elf");
        test_compress_elf(RETH_ELF);
    }
}
