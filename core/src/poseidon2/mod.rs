pub mod constants;

pub mod poseidon2_bb31_16_kernels {
    use p3_baby_bear::BabyBear;

    pub const ROUNDS_F: usize = 8;
    pub const ROUNDS_P: usize = 13;
    pub const WIDTH: usize = 16;
    pub const RATE: usize = 8;
    pub const DIGEST_WIDTH: usize = 8;
    pub const D_U64: u64 = 7;

    #[allow(unused_attributes)]
    #[link_name = "poseidon2_bb31_16_gpu"]
    extern "C" {
        pub fn permute_bb31(
            input: *const [BabyBear; WIDTH],
            output: *mut [BabyBear; WIDTH],
            n: usize,
            n_blocks: usize,
            n_threads_per_block: usize,
        );

        pub fn compress_bb31(
            left: *const [BabyBear; DIGEST_WIDTH],
            right: *const [BabyBear; DIGEST_WIDTH],
            output: *mut [BabyBear; DIGEST_WIDTH],
            n: usize,
            n_blocks: usize,
            n_threads_per_block: usize,
        );

        pub fn hash_bb31(
            input: *const BabyBear,
            n_input: usize,
            output: *mut [BabyBear; DIGEST_WIDTH],
            n: usize,
            n_blocks: usize,
            n_threads_per_block: usize,
        );
    }
}

pub mod poseidon2_bn254_3_kernels {
    use p3_bn254_fr::Bn254Fr;

    pub const DIGEST_WIDTH: usize = 1;
    pub const RATE: usize = 1;
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

// pub mod sum_bb31_kernels {
//     use super::poseidon2_bb31_16_kernels::WIDTH;
//     use p3_baby_bear::BabyBear;

//     #[allow(unused_attributes)]
//     #[link_name = "sum_bb31_gpu"]
//     extern "C" {
//         pub fn sum_bb31(
//             left: *const [BabyBear; WIDTH],
//             right: *const [BabyBear; WIDTH],
//             output: *mut [BabyBear; WIDTH],
//             n: usize,
//             n_blocks: usize,
//             n_threads_per_block: usize,
//         );
//     }
// }

// pub mod sum_bn254_kernels {
//     use super::poseidon2_bb31_16_kernels::WIDTH;
//     use p3_bn254_fr::Bn254Fr;

//     #[allow(unused_attributes)]
//     #[link_name = "sum_bb31_gpu"]
//     extern "C" {
//         pub fn sum_bn254(
//             left: *const [Bn254Fr; WIDTH],
//             right: *const [Bn254Fr; WIDTH],
//             output: *mut [Bn254Fr; WIDTH],
//             n: usize,
//             n_blocks: usize,
//             n_threads_per_block: usize,
//         );
//     }
// }

pub mod tests {
    #[cfg(test)]
    pub mod bb31_tests {

        use crate::device::buffer::DeviceBuffer;
        use crate::device::memory::ToDevice;
        use crate::device::memory::ToHost;
        use crate::poseidon2::constants::RC_16_30;
        use p3_baby_bear::BabyBear;
        use p3_baby_bear::DiffusionMatrixBabyBear;
        use p3_field::AbstractField;
        use p3_field::PrimeField32;
        use p3_poseidon2::Poseidon2;
        use p3_poseidon2::Poseidon2ExternalMatrixGeneral;
        use p3_symmetric::CryptographicHasher;
        use p3_symmetric::PaddingFreeSponge;
        use p3_symmetric::Permutation;
        use p3_symmetric::TruncatedPermutation;
        use rand::thread_rng;
        use rand::Rng;

        use super::super::poseidon2_bb31_16_kernels;
        use super::super::poseidon2_bb31_16_kernels::DIGEST_WIDTH;
        use super::super::poseidon2_bb31_16_kernels::D_U64;
        use super::super::poseidon2_bb31_16_kernels::RATE;
        use super::super::poseidon2_bb31_16_kernels::ROUNDS_F;
        use super::super::poseidon2_bb31_16_kernels::ROUNDS_P;
        use super::super::poseidon2_bb31_16_kernels::WIDTH;

        fn round_constants() -> (Vec<[BabyBear; 16]>, Vec<BabyBear>) {
            let mut round_constants = RC_16_30.to_vec();
            let internal_start = ROUNDS_F / 2;
            let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
            let internal_round_constants = round_constants
                .drain(internal_start..internal_end)
                .map(|vec| vec[0])
                .collect::<Vec<_>>();
            let external_round_constants = round_constants;
            (
                external_round_constants[0..ROUNDS_F].to_vec(),
                internal_round_constants,
            )
        }

        pub fn poseidon2_bb31_16_perm(
        ) -> Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>
        {
            let (external_round_constants, internal_round_constants) = round_constants();
            Poseidon2::<
                BabyBear,
                Poseidon2ExternalMatrixGeneral,
                DiffusionMatrixBabyBear,
                WIDTH,
                D_U64,
            >::new(
                ROUNDS_F,
                external_round_constants,
                Poseidon2ExternalMatrixGeneral,
                ROUNDS_P,
                internal_round_constants,
                DiffusionMatrixBabyBear,
            )
        }

        pub fn poseidon2_bb31_16_hasher() -> PaddingFreeSponge<
            Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>,
            WIDTH,
            RATE,
            DIGEST_WIDTH,
        > {
            let perm = poseidon2_bb31_16_perm();
            PaddingFreeSponge::new(perm)
        }

        pub fn poseidon2_bb31_16_compressor() -> TruncatedPermutation<
            Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>,
            2,
            8,
            16,
        > {
            let perm = poseidon2_bb31_16_perm();
            TruncatedPermutation::new(perm)
        }

        #[test]
        #[ignore]
        fn test_codegen_round_constants_gpu() {
            let (external_round_constants, internal_round_constants) = round_constants();

            let mut lines = Vec::new();
            lines.push(format!(
                "__constant__ bb31_t EXTERNAL_ROUND_CONSTANTS[{}][{}] = {{",
                external_round_constants.len(),
                external_round_constants[0].len()
            ));
            for (i, round_constants) in external_round_constants.iter().enumerate() {
                let mut line = "{".to_string()
                    + &round_constants
                        .iter()
                        .map(|rc| format!("bb31_t({})", rc.as_canonical_u32()))
                        .collect::<Vec<_>>()
                        .join(",")
                    + "}";
                if i != external_round_constants.len() - 1 {
                    line += ",";
                }
                lines.push(line);
            }
            lines.push("};".to_string());

            lines.push("".to_string());
            lines.push(format!(
                "__constant__ bb31_t INTERNAL_ROUND_CONSTANTS[{}] = {{",
                internal_round_constants.len()
            ));
            lines.push(
                internal_round_constants
                    .iter()
                    .map(|rc| format!("bb31_t({})", rc.as_canonical_u32()))
                    .collect::<Vec<_>>()
                    .join(","),
            );
            lines.push("};".to_string());

            println!("{}", lines.join("\n"));
        }

        #[test]
        fn test_permute_gpu() {
            // Setup the random number generator.
            let mut rng = thread_rng();

            // Setup the testing parameters.
            let n = 128;
            let threads_per_block = 32;
            let num_blocks = n / threads_per_block + 1;

            // Generate the input data on the host.
            let input = (0..n)
                .map(|_| [rng.gen::<BabyBear>(); WIDTH])
                .collect::<Vec<_>>();

            // Copy the input data to the device.
            let input_device = input.to_device();
            let mut output_device = DeviceBuffer::with_capacity(n * DIGEST_WIDTH);

            // Execute the source implementation.
            let perm = poseidon2_bb31_16_perm();
            let mut gt = Vec::new();
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let state: [BabyBear; WIDTH] = input[i];
                gt.push(perm.permute(state));
            }

            // Execute the kernel.
            unsafe {
                output_device.set_len(n * DIGEST_WIDTH);
                poseidon2_bb31_16_kernels::permute_bb31(
                    input_device.as_ptr(),
                    output_device.as_mut_ptr(),
                    n,
                    num_blocks,
                    threads_per_block,
                );
            }

            // Copy the result of the kernel to the host.
            let output = output_device.to_host();
            for i in 0..n {
                assert_eq!(gt[i], output[i]);
            }
        }

        #[test]
        fn test_compress_gpu() {
            // Setup the random number generator.
            let mut rng = thread_rng();

            // Setup the testing parameters.
            let n = 128;
            let threads_per_block = 32;
            let num_blocks = n / threads_per_block + 1;

            // Generate the input data on the host.
            let left = (0..n)
                .map(|_| [rng.gen::<BabyBear>(); DIGEST_WIDTH])
                .collect::<Vec<_>>();
            let right = (0..n)
                .map(|_| [rng.gen::<BabyBear>(); DIGEST_WIDTH])
                .collect::<Vec<_>>();
            let mut output: Vec<[BabyBear; DIGEST_WIDTH]> = Vec::new();
            output.resize(n, [BabyBear::zero(); DIGEST_WIDTH]);

            // Copy the input data to the device.
            let left_device = left.to_device();
            let right_device = right.to_device();
            let mut output_device = output.to_device();

            // Execute the source implementation.
            let perm = poseidon2_bb31_16_perm();
            let mut gt: Vec<[BabyBear; DIGEST_WIDTH]> = Vec::new();
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let mut state = [BabyBear::zero(); WIDTH];
                #[allow(clippy::manual_memcpy)]
                for j in 0..DIGEST_WIDTH {
                    state[j] = left[i][j];
                    state[j + DIGEST_WIDTH] = right[i][j];
                }
                gt.push(perm.permute(state)[0..DIGEST_WIDTH].try_into().unwrap());
            }

            // Execute the kernel.
            unsafe {
                poseidon2_bb31_16_kernels::compress_bb31(
                    left_device.as_ptr(),
                    right_device.as_ptr(),
                    output_device.as_slice_mut().as_mut_ptr(),
                    n,
                    num_blocks,
                    threads_per_block,
                );
            }

            // Copy the result of the kernel to the host.
            output_device.copy_to_host(&mut output[..]);
            for i in 0..n {
                assert_eq!(gt[i], output[i]);
            }
        }

        #[test]
        fn test_hash_gpu() {
            // Setup the random number generator.
            let mut rng = thread_rng();

            // Setup the testing parameters.
            let n = 128;
            const N_INPUT: usize = 107;
            let threads_per_block = 32;
            let num_blocks = n / threads_per_block + 1;

            // Generate the input data on the host.
            let input = (0..n)
                .flat_map(|_| [rng.gen::<BabyBear>(); N_INPUT].to_vec())
                .collect::<Vec<_>>();
            let mut output: Vec<[BabyBear; DIGEST_WIDTH]> = Vec::new();
            output.resize(n, [BabyBear::zero(); DIGEST_WIDTH]);

            // Copy the input data to the device.
            let input_device = input.to_device();
            let mut output_device = output.to_device();

            // Execute the source implementation.
            let sponge = poseidon2_bb31_16_hasher();

            let mut gt: Vec<[BabyBear; DIGEST_WIDTH]> = Vec::new();
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let data = input[i * N_INPUT..(i + 1) * N_INPUT].to_vec();
                gt.push(sponge.hash_iter(data));
            }

            // Execute the kernel.
            unsafe {
                poseidon2_bb31_16_kernels::hash_bb31(
                    input_device.as_slice().as_ptr(),
                    N_INPUT,
                    output_device.as_slice_mut().as_mut_ptr(),
                    n,
                    num_blocks,
                    threads_per_block,
                );
            }

            // Copy the result of the kernel to the host.
            output_device.copy_to_host(&mut output[..]);
            for i in 0..n {
                assert_eq!(gt[i], output[i]);
            }
        }
    }

    #[cfg(test)]
    pub mod bn254_tests {
        use crate::device::memory::ToDevice;
        use p3_bn254_fr::{Bn254Fr, DiffusionMatrixBN254};
        use p3_field::AbstractField;
        use p3_poseidon2::Poseidon2;
        use p3_poseidon2::Poseidon2ExternalMatrixGeneral;
        use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};
        use rand::thread_rng;
        use rand::Rng;
        // use sp1_core::utils::ec::weierstrass::bn254::Bn254;
        use sp1_recursion_core::stark::poseidon2::bn254_poseidon2_rc3;

        use super::super::poseidon2_bn254_3_kernels;
        use super::super::poseidon2_bn254_3_kernels::DIGEST_WIDTH;
        use super::super::poseidon2_bn254_3_kernels::D_U64;
        use super::super::poseidon2_bn254_3_kernels::RATE;
        use super::super::poseidon2_bn254_3_kernels::ROUNDS_F;
        use super::super::poseidon2_bn254_3_kernels::ROUNDS_P;
        use super::super::poseidon2_bn254_3_kernels::WIDTH;

        pub fn poseidon2_bn254_3_perm(
        ) -> Poseidon2<Bn254Fr, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBN254, 3, 5>
        {
            let mut round_constants = bn254_poseidon2_rc3();
            let internal_start = ROUNDS_F / 2;
            let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
            let internal_round_constants = round_constants
                .drain(internal_start..internal_end)
                .map(|vec| vec[0])
                .collect::<Vec<_>>();
            let external_round_constants = round_constants;
            Poseidon2::<Bn254Fr, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBN254, WIDTH, D_U64>::new(
                ROUNDS_F,
                external_round_constants,
                Poseidon2ExternalMatrixGeneral,
                ROUNDS_P,
                internal_round_constants,
                DiffusionMatrixBN254,
            )
        }

        pub fn poseidon2_bn254_3_hasher() -> PaddingFreeSponge<
            Poseidon2<Bn254Fr, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBN254, 3, 5>,
            WIDTH,
            RATE,
            DIGEST_WIDTH,
        > {
            let perm = poseidon2_bn254_3_perm();
            PaddingFreeSponge::new(perm)
        }

        #[test]
        fn test_hash_gpu_bn254() {
            // Setup the random number generator.
            let mut rng = thread_rng();

            // Setup the testing parameters.
            let n = 128;
            const N_INPUT: usize = 107;
            let threads_per_block = 32;
            let num_blocks = n / threads_per_block + 1;

            // Generate the input data on the host.
            let input = (0..n)
                .flat_map(|_| [rng.gen::<Bn254Fr>(); N_INPUT].to_vec())
                .collect::<Vec<_>>();
            let mut output: Vec<[Bn254Fr; DIGEST_WIDTH]> = Vec::new();
            output.resize(n, [Bn254Fr::zero(); DIGEST_WIDTH]);

            // Copy the input data to the device.
            let input_device = input.to_device();
            let mut output_device = output.to_device();

            // Execute the source implementation.
            let sponge = poseidon2_bn254_3_hasher();

            let mut round_constants = bn254_poseidon2_rc3();
            let internal_start = ROUNDS_F / 2;
            let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
            let internal_round_constants = round_constants
                .drain(internal_start..internal_end)
                .map(|vec| vec[0])
                .collect::<Vec<_>>();
            let external_round_constants = round_constants;

            let internal_rounds_constats_device = internal_round_constants.to_device();
            let external_rounds_constats_device = external_round_constants.to_device();
            let diffusion_matrix_m1 = [Bn254Fr::one(), Bn254Fr::one(), Bn254Fr::two()];
            let diffusion_matrix_m1_device = diffusion_matrix_m1.to_device();

            let mut gt: Vec<[Bn254Fr; DIGEST_WIDTH]> = Vec::new();
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let data = input[i * N_INPUT..(i + 1) * N_INPUT].to_vec();
                gt.push(sponge.hash_iter(data));
            }

            // Execute the kernel.
            unsafe {
                poseidon2_bn254_3_kernels::hash_bn254(
                    input_device.as_slice().as_ptr(),
                    N_INPUT,
                    output_device.as_slice_mut().as_mut_ptr(),
                    internal_rounds_constats_device.as_slice().as_ptr(),
                    external_rounds_constats_device.as_slice().as_ptr(),
                    diffusion_matrix_m1_device.as_slice().as_ptr(),
                    n,
                    num_blocks,
                    threads_per_block,
                );
            }

            // Copy the result of the kernel to the host.
            output_device.copy_to_host(&mut output[..]);
            for i in 0..n {
                assert_eq!(gt[i], output[i]);
            }
        }
    }

    // use super::sum_bb31_kernels;

    // #[test]
    // fn test_sum_bb31() {
    //     // Setup the random number generator.
    //     let mut rng = thread_rng();

    //     // Setup the testing parameters.
    //     let n = 128;
    //     let threads_per_block = 32;
    //     let num_blocks = n / threads_per_block + 1;

    //     // Generate the input data on the host.
    //     let left: Vec<[BabyBear; 16]> = (0..n)
    //         .map(|_| [rng.gen::<BabyBear>(); WIDTH])
    //         .collect::<Vec<_>>();
    //     let right: Vec<[BabyBear; 16]> = (0..n)
    //         .map(|_| [rng.gen::<BabyBear>(); WIDTH])
    //         .collect::<Vec<_>>();
    //     let mut output: Vec<[BabyBear; WIDTH]> = Vec::new();
    //     output.resize(n, [BabyBear::zero(); WIDTH]);

    //     // Copy the input data to the device.
    //     let left_device = left.to_device();
    //     let right_device = right.to_device();
    //     let mut output_device = output.to_device();

    //     // Execute the source implementation.
    //     let mut gt: Vec<[BabyBear; WIDTH]> = Vec::new();
    //     for i in 0..n {
    //         let mut state = [BabyBear::zero(); WIDTH];
    //         for j in 0..WIDTH {
    //             state[j] = left[i][j] + right[i][j];
    //         }
    //         gt.push(state);
    //     }

    //     // Execute the kernel.
    //     unsafe {
    //         sum_bb31_kernels::sum_bb31(
    //             left_device.as_ptr(),
    //             right_device.as_ptr(),
    //             output_device.as_slice_mut().as_mut_ptr(),
    //             n,
    //             num_blocks,
    //             threads_per_block,
    //         );
    //     }

    //     // Copy the result of the kernel to the host.
    //     output_device.copy_to_host(&mut output[..]);

    //     for i in 0..n {
    //         assert_eq!(gt[i], output[i]);
    //     }
    // }

    // use super::sum_bn254_kernels;
    // use p3_bn254_fr::Bn254Fr;

    // #[test]
    // fn test_sum_bn254() {
    //     // Setup the random number generator.
    //     let mut rng = thread_rng();

    //     // Setup the testing parameters.
    //     let n = 128;
    //     let threads_per_block = 32;
    //     let num_blocks = n / threads_per_block + 1;

    //     // Generate the input data on the host.
    //     let left: Vec<[Bn254Fr; 16]> = (0..n)
    //         .map(|_| [rng.gen::<Bn254Fr>(); WIDTH])
    //         .collect::<Vec<_>>();
    //     let right: Vec<[Bn254Fr; 16]> = (0..n)
    //         .map(|_| [rng.gen::<Bn254Fr>(); WIDTH])
    //         .collect::<Vec<_>>();
    //     let mut output: Vec<[Bn254Fr; WIDTH]> = Vec::new();
    //     output.resize(n, [Bn254Fr::zero(); WIDTH]);

    //     // Copy the input data to the device.
    //     let left_device = left.to_device();
    //     let right_device = right.to_device();
    //     let mut output_device = output.to_device();

    //     let M: Bn254Fr = Bn254Fr::one();

    //     // Execute the source implementation.
    //     let mut gt: Vec<[Bn254Fr; WIDTH]> = Vec::new();
    //     for i in 0..n {
    //         let mut state = [Bn254Fr::zero(); WIDTH];
    //         for j in 0..WIDTH {
    //             state[j] = left[i][j] + right[i][j];
    //             state[j] *= M;
    //         }
    //         gt.push(state);
    //     }

    //     // Execute the kernel.
    //     unsafe {
    //         sum_bn254_kernels::sum_bn254(
    //             left_device.as_ptr(),
    //             right_device.as_ptr(),
    //             output_device.as_slice_mut().as_mut_ptr(),
    //             n,
    //             num_blocks,
    //             threads_per_block,
    //         );
    //     }

    //     // Copy the result of the kernel to the host.
    //     output_device.copy_to_host(&mut output[..]);

    //     for i in 0..n {
    //         assert_eq!(gt[i], output[i]);
    //     }
    // }
}
