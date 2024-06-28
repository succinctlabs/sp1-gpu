use p3_baby_bear::BabyBear;

pub mod poseidon2_baby_bear_16_kernels {
    use p3_baby_bear::BabyBear;

    pub const ROUNDS_F: usize = 8;
    pub const ROUNDS_P: usize = 13;
    pub const WIDTH: usize = 16;
    pub const RATE: usize = 8;
    pub const DIGEST_WIDTH: usize = 8;
    pub const D_U64: u64 = 7;

    #[allow(unused_attributes)]
    #[link_name = "poseidon2_baby_bear_16_gpu"]
    extern "C" {
        pub fn permute_baby_bear(
            input: *const [BabyBear; WIDTH],
            output: *mut [BabyBear; WIDTH],
            n: usize,
            n_blocks: usize,
            n_threads_per_block: usize,
        );

        pub fn compress_baby_bear(
            left: *const [BabyBear; DIGEST_WIDTH],
            right: *const [BabyBear; DIGEST_WIDTH],
            output: *mut [BabyBear; DIGEST_WIDTH],
            n: usize,
            n_blocks: usize,
            n_threads_per_block: usize,
        );

        pub fn hash_baby_bear(
            input: *const BabyBear,
            n_input: usize,
            output: *mut [BabyBear; DIGEST_WIDTH],
            n: usize,
            n_blocks: usize,
            n_threads_per_block: usize,
        );
    }
}

use poseidon2_baby_bear_16_kernels::DIGEST_WIDTH;
use poseidon2_baby_bear_16_kernels::WIDTH;

pub struct HasherBabyBearGPU {}

impl Default for HasherBabyBearGPU {
    fn default() -> Self {
        HasherBabyBearGPU::new()
    }
}

impl HasherBabyBearGPU {
    pub fn new() -> Self {
        Self {}
    }

    /// # Safety
    pub unsafe fn permute(
        &self,
        input: *const [BabyBear; WIDTH],
        output: *mut [BabyBear; WIDTH],
        n: usize,
        n_blocks: usize,
        n_threads_per_block: usize,
    ) {
        poseidon2_baby_bear_16_kernels::permute_baby_bear(
            input,
            output,
            n,
            n_blocks,
            n_threads_per_block,
        );
    }

    /// # Safety
    pub unsafe fn compress(
        &self,
        left: *const [BabyBear; DIGEST_WIDTH],
        right: *const [BabyBear; DIGEST_WIDTH],
        output: *mut [BabyBear; DIGEST_WIDTH],
        n: usize,
        n_blocks: usize,
        n_threads_per_block: usize,
    ) {
        unsafe {
            poseidon2_baby_bear_16_kernels::compress_baby_bear(
                left,
                right,
                output,
                n,
                n_blocks,
                n_threads_per_block,
            );
        }
    }

    /// # Safety
    pub unsafe fn hash(
        &self,
        input: *const BabyBear,
        n_input: usize,
        output: *mut [BabyBear; DIGEST_WIDTH],
        n: usize,
        n_blocks: usize,
        n_threads_per_block: usize,
    ) {
        unsafe {
            poseidon2_baby_bear_16_kernels::hash_baby_bear(
                input,
                n_input,
                output,
                n,
                n_blocks,
                n_threads_per_block,
            );
        }
    }
}
