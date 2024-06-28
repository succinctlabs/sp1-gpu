use crate::device::buffer::DeviceBuffer;
use crate::device::memory::ToDevice;
use p3_bn254_fr::Bn254Fr;
use p3_field::AbstractField;

pub mod poseidon2_bn254_3_kernels {
    use p3_bn254_fr::Bn254Fr;

    pub const DIGEST_WIDTH: usize = 1;
    pub const RATE: usize = 2;
    pub const WIDTH: usize = 3;
    pub const ROUNDS_F: usize = 8;
    pub const ROUNDS_P: usize = 56;
    pub const D_U64: u64 = 5;

    #[allow(unused_attributes)]
    #[link_name = "poseidon2_bn254_3_gpu"]
    extern "C" {
        pub fn permute_bn254(
            input: *const [Bn254Fr; WIDTH],
            output: *mut [Bn254Fr; WIDTH],
            internal_round_constants: *const Bn254Fr,
            external_round_constants: *const [Bn254Fr; WIDTH],
            diffusion_matrix_m1: *const Bn254Fr,
            n: usize,
            n_blocks: usize,
            n_threads_per_block: usize,
        );

        pub fn compress_bn254(
            left: *const [Bn254Fr; DIGEST_WIDTH],
            right: *const [Bn254Fr; DIGEST_WIDTH],
            output: *mut [Bn254Fr; DIGEST_WIDTH],
            internal_round_constants: *const Bn254Fr,
            external_round_constants: *const [Bn254Fr; WIDTH],
            diffusion_matrix_m1: *const Bn254Fr,
            n: usize,
            n_blocks: usize,
            n_threads_per_block: usize,
        );

        pub fn hash_bn254(
            input: *const Bn254Fr,
            n_input: usize,
            output: *mut [Bn254Fr; DIGEST_WIDTH],
            internal_round_constants: *const Bn254Fr,
            external_round_constants: *const [Bn254Fr; WIDTH],
            diffusion_matrix_m1: *const Bn254Fr,
            n: usize,
            n_blocks: usize,
            n_threads_per_block: usize,
        );
    }
}

use poseidon2_bn254_3_kernels::DIGEST_WIDTH;
use poseidon2_bn254_3_kernels::ROUNDS_F;
use poseidon2_bn254_3_kernels::ROUNDS_P;
use poseidon2_bn254_3_kernels::WIDTH;
use sp1_recursion_core::stark::poseidon2::bn254_poseidon2_rc3;

pub fn poseidon2_bn254_3_constants() -> (Vec<Bn254Fr>, Vec<[Bn254Fr; WIDTH]>, Vec<Bn254Fr>) {
    let mut round_constants = bn254_poseidon2_rc3();
    let internal_start = ROUNDS_F / 2;
    let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
    let internal_round_constants = round_constants
        .drain(internal_start..internal_end)
        .map(|vec| vec[0])
        .collect::<Vec<_>>();
    let external_round_constants = round_constants;
    let diffusion_matrix_m1 = [Bn254Fr::one(), Bn254Fr::one(), Bn254Fr::two()].to_vec();
    (
        internal_round_constants,
        external_round_constants,
        diffusion_matrix_m1,
    )
}

pub struct HasherBn254GPU {
    internal_rounds_constats_device: DeviceBuffer<Bn254Fr>,
    external_rounds_constats_device: DeviceBuffer<[Bn254Fr; WIDTH]>,
    diffusion_matrix_m1_device: DeviceBuffer<Bn254Fr>,
}

impl Default for HasherBn254GPU {
    fn default() -> Self {
        HasherBn254GPU::new()
    }
}

impl HasherBn254GPU {
    pub fn new() -> Self {
        let (internal_rounds_constats, external_rounds_constats, diffusion_matrix_m1) =
            poseidon2_bn254_3_constants();
        Self {
            internal_rounds_constats_device: internal_rounds_constats.to_device(),
            external_rounds_constats_device: external_rounds_constats.to_device(),
            diffusion_matrix_m1_device: diffusion_matrix_m1.to_device(),
        }
    }

    /// # Safety
    pub unsafe fn permute(
        self,
        input: *const [Bn254Fr; WIDTH],
        output: *mut [Bn254Fr; WIDTH],
        n: usize,
        n_blocks: usize,
        n_threads_per_block: usize,
    ) {
        unsafe {
            poseidon2_bn254_3_kernels::permute_bn254(
                input,
                output,
                self.internal_rounds_constats_device.as_slice().as_ptr(),
                self.external_rounds_constats_device.as_slice().as_ptr(),
                self.diffusion_matrix_m1_device.as_slice().as_ptr(),
                n,
                n_blocks,
                n_threads_per_block,
            );
        }
    }

    /// # Safety
    pub unsafe fn compress(
        &self,
        left: *const [Bn254Fr; DIGEST_WIDTH],
        right: *const [Bn254Fr; DIGEST_WIDTH],
        output: *mut [Bn254Fr; DIGEST_WIDTH],
        n: usize,
        n_blocks: usize,
        n_threads_per_block: usize,
    ) {
        unsafe {
            poseidon2_bn254_3_kernels::compress_bn254(
                left,
                right,
                output,
                self.internal_rounds_constats_device.as_slice().as_ptr(),
                self.external_rounds_constats_device.as_slice().as_ptr(),
                self.diffusion_matrix_m1_device.as_slice().as_ptr(),
                n,
                n_blocks,
                n_threads_per_block,
            );
        }
    }

    /// # Safety
    pub unsafe fn hash(
        &self,
        input: *const Bn254Fr,
        n_input: usize,
        output: *mut [Bn254Fr; DIGEST_WIDTH],
        n: usize,
        n_blocks: usize,
        n_threads_per_block: usize,
    ) {
        unsafe {
            poseidon2_bn254_3_kernels::hash_bn254(
                input,
                n_input,
                output,
                self.internal_rounds_constats_device.as_slice().as_ptr(),
                self.external_rounds_constats_device.as_slice().as_ptr(),
                self.diffusion_matrix_m1_device.as_slice().as_ptr(),
                n,
                n_blocks,
                n_threads_per_block,
            );
        }
    }
}
