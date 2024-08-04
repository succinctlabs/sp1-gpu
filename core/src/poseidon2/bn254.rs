use crate::device::memory::ToDevice;
use crate::device::DeviceBuffer;
use crate::matrix::MatrixViewDevice;
use crate::merkle_tree::FieldMerkleTreeHasher;
use p3_baby_bear::BabyBear;
use p3_bn254_fr::Bn254Fr;
use p3_field::AbstractField;

pub struct DeviceHasherBn254 {
    internal_rounds_constats_device: DeviceBuffer<Bn254Fr>,
    external_rounds_constats_device: DeviceBuffer<[Bn254Fr; WIDTH]>,
    diffusion_matrix_m1_device: DeviceBuffer<Bn254Fr>,
}

impl FieldMerkleTreeHasher<BabyBear> for DeviceHasherBn254 {
    type Digest = [Bn254Fr; DIGEST_WIDTH];

    unsafe fn first_digest_layer(
        &self,
        tallest_matrices: *const MatrixViewDevice<BabyBear>,
        n_tallest_matrices: usize,
        digests: *mut Self::Digest,
        max_height: usize,
    ) {
        poseidon2_bn254_3_kernels::first_digest_layer_bn254(
            tallest_matrices,
            n_tallest_matrices,
            digests,
            self.internal_rounds_constats_device.as_ptr(),
            self.external_rounds_constats_device.as_ptr(),
            self.diffusion_matrix_m1_device.as_ptr(),
            max_height,
        )
    }

    unsafe fn compress_and_inject(
        &self,
        prev_layer: *const Self::Digest,
        matrices_to_inject: *const MatrixViewDevice<BabyBear>,
        n_matrices_to_inject: usize,
        next_digests: *mut Self::Digest,
        layer_len: usize,
    ) {
        poseidon2_bn254_3_kernels::compress_and_inject_bn254(
            prev_layer,
            matrices_to_inject,
            n_matrices_to_inject,
            next_digests,
            self.internal_rounds_constats_device.as_ptr(),
            self.external_rounds_constats_device.as_ptr(),
            self.diffusion_matrix_m1_device.as_ptr(),
            layer_len,
        );
    }
}

impl Default for DeviceHasherBn254 {
    fn default() -> Self {
        DeviceHasherBn254::new()
    }
}

impl DeviceHasherBn254 {
    pub fn new() -> Self {
        let (internal_rounds_constats, external_rounds_constats, diffusion_matrix_m1) =
            poseidon2_bn254_3_constants();
        Self {
            internal_rounds_constats_device: internal_rounds_constats.to_device().unwrap(),
            external_rounds_constats_device: external_rounds_constats.to_device().unwrap(),
            diffusion_matrix_m1_device: diffusion_matrix_m1.to_device().unwrap(),
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
        poseidon2_bn254_3_kernels::permute_bn254(
            input,
            output,
            self.internal_rounds_constats_device.as_ptr(),
            self.external_rounds_constats_device.as_ptr(),
            self.diffusion_matrix_m1_device.as_ptr(),
            n,
            n_blocks,
            n_threads_per_block,
        );
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
        poseidon2_bn254_3_kernels::compress_bn254(
            left,
            right,
            output,
            self.internal_rounds_constats_device.as_ptr(),
            self.external_rounds_constats_device.as_ptr(),
            self.diffusion_matrix_m1_device.as_ptr(),
            n,
            n_blocks,
            n_threads_per_block,
        );
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
        poseidon2_bn254_3_kernels::hash_bn254(
            input,
            n_input,
            output,
            self.internal_rounds_constats_device.as_ptr(),
            self.external_rounds_constats_device.as_ptr(),
            self.diffusion_matrix_m1_device.as_ptr(),
            n,
            n_blocks,
            n_threads_per_block,
        );
    }
}

pub mod poseidon2_bn254_3_kernels {
    use crate::matrix::MatrixViewDevice;
    use p3_baby_bear::BabyBear;
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

    #[allow(unused_attributes)]
    #[link_name = "merkle_tree_bn254_16_gpu"]
    extern "C" {
        pub fn first_digest_layer_bn254(
            tallest_matrices: *const MatrixViewDevice<BabyBear>,
            n_tallest_matrices: usize,
            digests: *mut [Bn254Fr; DIGEST_WIDTH],
            internal_round_constants: *const Bn254Fr,
            external_round_constants: *const [Bn254Fr; WIDTH],
            diffusion_matrix_m1: *const Bn254Fr,
            max_height: usize,
        );

        pub fn compress_and_inject_bn254(
            prev_layer: *const [Bn254Fr; DIGEST_WIDTH],
            //n_prev_layer: usize,
            matrices_to_inject: *const MatrixViewDevice<BabyBear>,
            n_matrices_to_inject: usize,
            next_digests: *mut [Bn254Fr; DIGEST_WIDTH],
            internal_round_constants: *const Bn254Fr,
            external_round_constants: *const [Bn254Fr; WIDTH],
            diffusion_matrix_m1: *const Bn254Fr,
            max_height: usize,
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
