pub mod constants;
pub mod hashers;

pub mod tests {
    #[cfg(test)]
    pub mod baby_bear_tests {

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

        use crate::poseidon2::hashers::baby_bear_16::HasherBabyBearGPU;
        use crate::poseidon2::hashers::poseidon2_baby_bear_16_kernels::DIGEST_WIDTH;
        use crate::poseidon2::hashers::poseidon2_baby_bear_16_kernels::D_U64;
        use crate::poseidon2::hashers::poseidon2_baby_bear_16_kernels::RATE;
        use crate::poseidon2::hashers::poseidon2_baby_bear_16_kernels::ROUNDS_F;
        use crate::poseidon2::hashers::poseidon2_baby_bear_16_kernels::ROUNDS_P;
        use crate::poseidon2::hashers::poseidon2_baby_bear_16_kernels::WIDTH;

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

        pub fn poseidon2_baby_bear_16_perm(
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

        pub fn poseidon2_baby_bear_16_hasher() -> PaddingFreeSponge<
            Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>,
            WIDTH,
            RATE,
            DIGEST_WIDTH,
        > {
            let perm = poseidon2_baby_bear_16_perm();
            PaddingFreeSponge::new(perm)
        }

        pub fn poseidon2_baby_bear_16_compressor() -> TruncatedPermutation<
            Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>,
            2,
            8,
            16,
        > {
            let perm = poseidon2_baby_bear_16_perm();
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
        fn test_permute_baby_bear_gpu() {
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
            let perm = poseidon2_baby_bear_16_perm();
            let mut gt = Vec::new();
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let state: [BabyBear; WIDTH] = input[i];
                gt.push(perm.permute(state));
            }

            // Execute the kernel.
            let hasher = HasherBabyBearGPU::new();
            unsafe {
                output_device.set_len(n * DIGEST_WIDTH);
                hasher.permute(
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
        fn test_compress_baby_bear_gpu() {
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
            let perm = poseidon2_baby_bear_16_perm();
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
            let hasher = HasherBabyBearGPU::new();
            unsafe {
                hasher.compress(
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
        fn test_hash_baby_bear_gpu() {
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
            let sponge = poseidon2_baby_bear_16_hasher();

            let mut gt: Vec<[BabyBear; DIGEST_WIDTH]> = Vec::new();
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let data = input[i * N_INPUT..(i + 1) * N_INPUT].to_vec();
                gt.push(sponge.hash_iter(data));
            }

            // Execute the kernel.
            let hasher = HasherBabyBearGPU::new();
            unsafe {
                hasher.hash(
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
        use crate::device::buffer::DeviceBuffer;
        use crate::device::memory::ToDevice;
        use crate::device::memory::ToHost;
        use p3_bn254_fr::{Bn254Fr, DiffusionMatrixBN254};
        use p3_field::AbstractField;
        use p3_poseidon2::Poseidon2;
        use p3_poseidon2::Poseidon2ExternalMatrixGeneral;
        use p3_symmetric::Permutation;
        use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};
        use rand::thread_rng;
        use rand::Rng;

        use crate::poseidon2::hashers::bn254_3::poseidon2_bn254_3_constants;
        use crate::poseidon2::hashers::bn254_3::HasherBn254GPU;
        use crate::poseidon2::hashers::poseidon2_bn254_3_kernels::DIGEST_WIDTH;
        use crate::poseidon2::hashers::poseidon2_bn254_3_kernels::D_U64;
        use crate::poseidon2::hashers::poseidon2_bn254_3_kernels::RATE;
        use crate::poseidon2::hashers::poseidon2_bn254_3_kernels::ROUNDS_F;
        use crate::poseidon2::hashers::poseidon2_bn254_3_kernels::ROUNDS_P;
        use crate::poseidon2::hashers::poseidon2_bn254_3_kernels::WIDTH;

        pub fn poseidon2_bn254_3_perm(
        ) -> Poseidon2<Bn254Fr, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBN254, 3, 5>
        {
            let (internal_round_constants, external_round_constants, _) =
                poseidon2_bn254_3_constants();
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

        // pub fn poseidon2_bn254_3_compressor() -> TruncatedPermutation<
        //     Poseidon2<Bn254Fr, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBN254, 3, 5>,
        //     2,
        //     1,
        //     3,
        // > {
        //     let perm = poseidon2_bn254_3_perm();
        //     TruncatedPermutation::new(perm)
        // }

        #[test]
        fn test_permute_bn254_gpu() {
            // Setup the random number generator.
            let mut rng = thread_rng();

            // Setup the testing parameters.
            let n = 128;
            let threads_per_block = 32;
            let num_blocks = n / threads_per_block + 1;

            // Generate the input data on the host.
            let input = (0..n)
                .map(|_| [rng.gen::<Bn254Fr>(); WIDTH])
                .collect::<Vec<_>>();

            // Copy the input data to the device.
            let input_device = input.to_device();
            let mut output_device = DeviceBuffer::with_capacity(n * DIGEST_WIDTH);

            // Execute the source implementation.
            let perm = poseidon2_bn254_3_perm();
            let mut gt = Vec::new();
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let state: [Bn254Fr; WIDTH] = input[i];
                gt.push(perm.permute(state));
            }

            // Execute the kernel.
            let hasher = HasherBn254GPU::new();
            unsafe {
                output_device.set_len(n * DIGEST_WIDTH);
                hasher.permute(
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
        fn test_compress_bn254_gpu() {
            // Setup the random number generator.
            let mut rng = thread_rng();

            // Setup the testing parameters.
            let n = 128;
            let threads_per_block = 32;
            let num_blocks = n / threads_per_block + 1;

            // Generate the input data on the host.
            let left = (0..n)
                .map(|_| [rng.gen::<Bn254Fr>(); DIGEST_WIDTH])
                .collect::<Vec<_>>();
            let right = (0..n)
                .map(|_| [rng.gen::<Bn254Fr>(); DIGEST_WIDTH])
                .collect::<Vec<_>>();
            let mut output: Vec<[Bn254Fr; DIGEST_WIDTH]> = Vec::new();
            output.resize(n, [Bn254Fr::zero(); DIGEST_WIDTH]);

            // Copy the input data to the device.
            let left_device = left.to_device();
            let right_device = right.to_device();
            let mut output_device = output.to_device();

            // Execute the source implementation.
            let perm = poseidon2_bn254_3_perm();
            let mut gt: Vec<[Bn254Fr; DIGEST_WIDTH]> = Vec::new();
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let mut state = [Bn254Fr::zero(); WIDTH];
                #[allow(clippy::manual_memcpy)]
                for j in 0..DIGEST_WIDTH {
                    state[j] = left[i][j];
                    state[j + DIGEST_WIDTH] = right[i][j];
                }
                gt.push(perm.permute(state)[0..DIGEST_WIDTH].try_into().unwrap());
            }

            // Execute the kernel.
            let hasher = HasherBn254GPU::new();
            unsafe {
                hasher.compress(
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
        fn test_hash_bn254_gpu() {
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
            let mut gt: Vec<[Bn254Fr; DIGEST_WIDTH]> = Vec::new();
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let data = input[i * N_INPUT..(i + 1) * N_INPUT].to_vec();
                gt.push(sponge.hash_iter(data));
            }

            // Execute the kernel.
            let hasher = HasherBn254GPU::new();
            unsafe {
                hasher.hash(
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
}
