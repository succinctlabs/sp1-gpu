pub mod duplex_challenger;

pub mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use rand::{thread_rng, Rng};
}

//     use crate::{
//         challenger::duplex_challenger::duplex_challenger_kernels::DIGEST_WIDTH,
//         poseidon2::baby_bear::DeviceHasherBabyBear,
//     };

//     #[test]
//     fn test_hash_baby_bear_gpu() {
//         // Setup the random number generator.
//         let mut rng = thread_rng();

//         // Setup the testing parameters.
//         let n = 128;
//         const N_INPUT: usize = 107;
//         let threads_per_block = 32;
//         let num_blocks = n / threads_per_block + 1;

//         // Generate the input data on the host.
//         let input =
//             (0..n).flat_map(|_| [rng.gen::<BabyBear>(); N_INPUT].to_vec()).collect::<Vec<_>>();
//         let mut output: Vec<[BabyBear; DIGEST_WIDTH]> = Vec::new();
//         output.resize(n, [BabyBear::zero(); DIGEST_WIDTH]);

//         // Copy the input data to the device.
//         let input_device = input.to_device().unwrap();
//         let mut output_device = output.to_device().unwrap();

//         // Execute the source implementation.
//         let sponge = poseidon2_baby_bear_16_hasher();

//         let mut gt: Vec<[BabyBear; DIGEST_WIDTH]> = Vec::new();
//         #[allow(clippy::needless_range_loop)]
//         for i in 0..n {
//             let data = input[i * N_INPUT..(i + 1) * N_INPUT].to_vec();
//             gt.push(sponge.hash_iter(data));
//         }

//         // Execute the kernel.
//         let hasher = DeviceHasherBabyBear::new();
//         unsafe {
//             hasher.hash(
//                 input_device.as_slice().as_ptr(),
//                 N_INPUT,
//                 output_device.as_slice_mut().as_mut_ptr(),
//                 n,
//                 num_blocks,
//                 threads_per_block,
//             );
//         }

//         // Copy the result of the kernel to the host.
//         output_device.copy_to_host(&mut output[..]);
//         for i in 0..n {
//             assert_eq!(gt[i], output[i]);
//         }
//     }
// }
