// pub mod merkle_tree_gpu {
//     use p3_baby_bear::BabyBear;

//     use crate::{
//         device::slice::DevicePtr,
//         poseidon2::poseidon2_bb31_16_gpu::{DIGEST_WIDTH, WIDTH},
//     };

//     #[derive(Clone, Copy)]
//     #[repr(C)]
//     pub struct RowMajorMatrixDevice {
//         pub data: DevicePtr<BabyBear>,
//         pub width: usize,
//         pub height: usize,
//     }

//     #[allow(unused_attributes)]
//     #[link_name = "merkle_tree_gpu"]
//     extern "C" {
//         #[link_name = "firstDigestLayer"]
//         pub fn first_digest_layer(
//             tallest_matrices: DevicePtr<RowMajorMatrixDevice>,
//             digests: DevicePtr<[BabyBear; DIGEST_WIDTH]>,
//             external_rc: DevicePtr<[BabyBear; WIDTH]>,
//             internal_rc: DevicePtr<BabyBear>,
//             n_blocks: usize,
//             n_threads_per_block: usize,
//         );

//         #[link_name = "compressAndInject"]
//         pub fn compress_and_inject(
//             prev_layer: DevicePtr<[BabyBear; WIDTH]>,
//             matrices_to_inject: DevicePtr<RowMajorMatrixDevice>,
//             next_digests: DevicePtr<[BabyBear; DIGEST_WIDTH]>,
//             external_rc: DevicePtr<[BabyBear; WIDTH]>,
//             internal_rc: DevicePtr<BabyBear>,
//             n_blocks: usize,
//             n_threads_per_block: usize,
//         );
//     }
// }

// #[cfg(test)]
// mod tests {
//     #[test]
//     fn test_first_digest_layer() {
//         println!("test");
//     }
// }
