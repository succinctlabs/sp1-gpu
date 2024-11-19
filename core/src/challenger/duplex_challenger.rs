// use crate::{
//     cuda_runtime::stream::CudaStreamHandle, matrix::MatrixViewDevice,
//     merkle_tree::FieldMerkleTreeHasher,
// };
// use p3_baby_bear::BabyBear;

// use p3_challenger::GrindingChallenger;
// use poseidon2_baby_bear_16_kernels::{DIGEST_WIDTH, WIDTH};

// #[derive(Clone)]
// pub struct DeviceDuplexChallenger {}

// impl GrindingChallenger for DeviceDuplexChallenger {
//     type Witness = BabyBear;

//     fn grind(&mut self, bits: usize) -> Self::Witness {
//         todo!()
//     }
// }
// pub mod poseidon2_baby_bear_16_kernels {
//     use crate::{cuda_runtime::stream::CudaStreamHandle, matrix::MatrixViewDevice};
//     use p3_baby_bear::BabyBear;

//     pub const ROUNDS_F: usize = 8;
//     pub const ROUNDS_P: usize = 13;
//     pub const WIDTH: usize = 16;
//     pub const RATE: usize = 8;
//     pub const DIGEST_WIDTH: usize = 8;
//     pub const D_U64: u64 = 7;

//     #[allow(unused_attributes)]
//     #[link_name = "poseidon2_baby_bear_16_gpu"]
//     extern "C" {
//         pub fn permute_baby_bear(
//             input: *const [BabyBear; WIDTH],
//             output: *mut [BabyBear; WIDTH],
//             n: usize,
//             n_blocks: usize,
//             n_threads_per_block: usize,
//         );

//         pub fn compress_baby_bear(
//             left: *const [BabyBear; DIGEST_WIDTH],
//             right: *const [BabyBear; DIGEST_WIDTH],
//             output: *mut [BabyBear; DIGEST_WIDTH],
//             n: usize,
//             n_blocks: usize,
//             n_threads_per_block: usize,
//         );

//         pub fn hash_baby_bear(
//             input: *const BabyBear,
//             n_input: usize,
//             output: *mut [BabyBear; DIGEST_WIDTH],
//             n: usize,
//             n_blocks: usize,
//             n_threads_per_block: usize,
//         );
//     }

//     #[allow(unused_attributes)]
//     #[link_name = "merkle_tree_baby_bear_16_gpu"]
//     extern "C" {
//         pub fn first_digest_layer_baby_bear(
//             tallest_matrices: *const MatrixViewDevice<BabyBear>,
//             n_tallest_matrices: usize,
//             digests: *mut [BabyBear; DIGEST_WIDTH],
//             max_height: usize,
//             stream: CudaStreamHandle,
//         );

//         pub fn compress_and_inject_baby_bear(
//             prev_layer: *const [BabyBear; DIGEST_WIDTH],
//             matrices_to_inject: *const MatrixViewDevice<BabyBear>,
//             n_matrices_to_inject: usize,
//             next_digests: *mut [BabyBear; DIGEST_WIDTH],
//             layer_len: usize,
//             stream: CudaStreamHandle,
//         );
//     }
// }

use p3_baby_bear::BabyBear;
use p3_challenger::GrindingChallenger;
use p3_field::{AbstractField, PrimeField64};
use sp1_recursion_core::stark::OuterChallenger;
use sp1_stark::InnerChallenger;

use crate::{
    cuda_runtime::stream::CudaStreamHandle,
    device::memory::{ToDevice, ToHost},
};

pub trait GrindOnDevice: GrindingChallenger {
    fn grind_on_device(&mut self, bits: usize, stream: CudaStreamHandle) -> Self::Witness;
}

impl GrindOnDevice for InnerChallenger {
    fn grind_on_device(&mut self, bits: usize, stream: CudaStreamHandle) -> Self::Witness {
        let mut result = BabyBear::zero();
        let mut sponge_state_device = self.sponge_state.to_device().unwrap();
        let mut input_buffer_device = self.input_buffer.to_device().unwrap();
        let mut output_buffer_device = self.output_buffer.to_device().unwrap();

        unsafe {
            grind(
                input_buffer_device.as_mut_ptr(),
                sponge_state_device.as_mut_ptr(),
                output_buffer_device.as_mut_ptr(),
                self.input_buffer.len(),
                self.output_buffer.len(),
                bits,
                BabyBear::ORDER_U64 as usize,
                &mut result,
                32,
                stream,
            );
        }

        self.input_buffer = input_buffer_device.to_host();
        self.sponge_state = sponge_state_device.to_host().try_into().unwrap();
        self.output_buffer = output_buffer_device.to_host();

        result
    }
}

impl GrindOnDevice for OuterChallenger {
    fn grind_on_device(&mut self, bits: usize, _stream: CudaStreamHandle) -> Self::Witness {
        self.grind(bits)
    }
}

#[allow(unused_attributes)]
#[link_name = "duplex_challenger"]
extern "C" {
    pub fn grind(
        input_buffer: *mut BabyBear,
        sponge_state: *mut BabyBear,
        output_buffer: *mut BabyBear,
        input_buffer_size: usize,
        output_buffer_size: usize,
        bits: usize,
        n: usize,
        out: *mut BabyBear,
        nThreadsPerBlock: usize,
        stream: CudaStreamHandle,
    );
}
