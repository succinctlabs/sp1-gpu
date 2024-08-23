pub const FIBONACCI_ELF: &[u8] =
    include_bytes!("../programs/fibonacci/riscv32im-succinct-zkvm-elf");

pub const LOOP_ELF: &[u8] = include_bytes!("../programs/loop/riscv32im-succinct-zkvm-elf");

pub const SHA2_CHAIN_ELF: &[u8] =
    include_bytes!("../programs/sha2-chain/riscv32im-succinct-zkvm-elf");

pub const TENDERMINT_BENCHMARK_ELF: &[u8] =
    include_bytes!("../programs/tendermint-benchmark/riscv32im-succinct-zkvm-elf");

pub const RETH_ELF: &[u8] = include_bytes!("../programs/reth/riscv32im-succinct-zkvm-elf");

pub const KEYSPACE_ELF: &[u8] = include_bytes!("../programs/keyspace/riscv32im-succinct-zkvm-elf");
pub const KEYSPACE_STDIN: &[u8] = include_bytes!("../programs/keyspace/stdin.bin");
