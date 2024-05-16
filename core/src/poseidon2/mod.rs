pub mod constants;

pub mod poseidon2_bb31_16_gpu {
    use p3_baby_bear::BabyBear;

    use crate::device::slice::DeviceSliceRaw;

    pub const ROUNDS_F: usize = 8;
    pub const ROUNDS_P: usize = 13;
    pub const WIDTH: usize = 16;
    pub const RATE: usize = 8;
    pub const DIGEST_WIDTH: usize = 8;
    pub const D_U64: u64 = 7;

    #[allow(unused_attributes)]
    #[link_name = "poseidon2_bb31_16_gpu"]
    extern "C" {
        pub fn permute(
            input: DeviceSliceRaw<[BabyBear; WIDTH]>,
            output: DeviceSliceRaw<[BabyBear; WIDTH]>,
            external_rc: DeviceSliceRaw<[BabyBear; WIDTH]>,
            internal_rc: DeviceSliceRaw<BabyBear>,
            n: usize,
            n_blocks: usize,
            n_threads_per_block: usize,
        );

        pub fn compress(
            left: DeviceSliceRaw<[BabyBear; DIGEST_WIDTH]>,
            right: DeviceSliceRaw<[BabyBear; DIGEST_WIDTH]>,
            output: DeviceSliceRaw<[BabyBear; DIGEST_WIDTH]>,
            external_rc: DeviceSliceRaw<[BabyBear; WIDTH]>,
            internal_rc: DeviceSliceRaw<BabyBear>,
            n: usize,
            n_blocks: usize,
            n_threads_per_block: usize,
        );

        pub fn hash(
            input: DeviceSliceRaw<BabyBear>,
            n_input: usize,
            output: DeviceSliceRaw<[BabyBear; DIGEST_WIDTH]>,
            external_rc: DeviceSliceRaw<[BabyBear; WIDTH]>,
            internal_rc: DeviceSliceRaw<BabyBear>,
            n: usize,
            n_blocks: usize,
            n_threads_per_block: usize,
        );
    }
}

#[cfg(test)]
mod tests {

    use crate::device::buffer::ToDevice;
    use crate::poseidon2::constants::RC_16_30;
    use p3_baby_bear::BabyBear;
    use p3_baby_bear::DiffusionMatrixBabyBear;
    use p3_field::AbstractField;
    use p3_poseidon2::Poseidon2;
    use p3_poseidon2::Poseidon2ExternalMatrixGeneral;
    use p3_symmetric::CryptographicHasher;
    use p3_symmetric::PaddingFreeSponge;
    use p3_symmetric::Permutation;
    use rand::thread_rng;
    use rand::Rng;

    use super::poseidon2_bb31_16_gpu;
    use super::poseidon2_bb31_16_gpu::DIGEST_WIDTH;
    use super::poseidon2_bb31_16_gpu::D_U64;
    use super::poseidon2_bb31_16_gpu::RATE;
    use super::poseidon2_bb31_16_gpu::ROUNDS_F;
    use super::poseidon2_bb31_16_gpu::ROUNDS_P;
    use super::poseidon2_bb31_16_gpu::WIDTH;

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

    fn perm() -> Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>
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

    #[test]
    fn test_permute_gpu() {
        // Setup the random number generator.
        let mut rng = thread_rng();

        // Setup the testing parameters.
        let n = 128;
        let threads_per_block = 32;
        let num_blocks = n / threads_per_block + 1;

        // Generate the input data on the host.
        let (external_round_constants, internal_round_constants) = round_constants();
        let input = (0..n)
            .map(|_| [rng.gen::<BabyBear>(); WIDTH])
            .collect::<Vec<_>>();
        let mut output: Vec<[BabyBear; WIDTH]> = Vec::new();
        output.resize(n, [BabyBear::zero(); WIDTH]);

        // Copy the input data to the device.
        let input_device = input.to_device();
        let output_device = output.to_device();
        let external_rc_device = external_round_constants.to_device();
        let internal_rc_device = internal_round_constants.to_device();

        // Execute the source implementation.
        let perm = perm();
        let mut gt = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let state: [BabyBear; WIDTH] = input[i];
            gt.push(perm.permute(state));
        }

        // Execute the kernel.
        unsafe {
            poseidon2_bb31_16_gpu::permute(
                input_device.as_slice().raw(),
                output_device.as_slice().raw(),
                external_rc_device.as_slice().raw(),
                internal_rc_device.as_slice().raw(),
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
    fn test_compress_gpu() {
        // Setup the random number generator.
        let mut rng = thread_rng();

        // Setup the testing parameters.
        let n = 128;
        let threads_per_block = 32;
        let num_blocks = n / threads_per_block + 1;

        // Generate the input data on the host.
        let (external_round_constants, internal_round_constants) = round_constants();
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
        let output_device = output.to_device();
        let external_rc_device = external_round_constants.to_device();
        let internal_rc_device = internal_round_constants.to_device();

        // Execute the source implementation.
        let perm = perm();
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
            poseidon2_bb31_16_gpu::compress(
                left_device.as_slice().raw(),
                right_device.as_slice().raw(),
                output_device.as_slice().raw(),
                external_rc_device.as_slice().raw(),
                internal_rc_device.as_slice().raw(),
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
        let (external_round_constants, internal_round_constants) = round_constants();
        let input = (0..n)
            .flat_map(|_| [rng.gen::<BabyBear>(); N_INPUT].to_vec())
            .collect::<Vec<_>>();
        let mut output: Vec<[BabyBear; DIGEST_WIDTH]> = Vec::new();
        output.resize(n, [BabyBear::zero(); DIGEST_WIDTH]);

        // Copy the input data to the device.
        let input_device = input.to_device();
        let output_device = output.to_device();
        let external_rc_device = external_round_constants.to_device();
        let internal_rc_device = internal_round_constants.to_device();

        // Execute the source implementation.
        let perm = perm();
        let sponge: PaddingFreeSponge<
            Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>,
            WIDTH,
            RATE,
            DIGEST_WIDTH,
        > = PaddingFreeSponge::new(perm);

        let mut gt: Vec<[BabyBear; DIGEST_WIDTH]> = Vec::new();
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let data = input[i * N_INPUT..(i + 1) * N_INPUT].to_vec();
            gt.push(sponge.hash_iter(data));
        }

        // Execute the kernel.
        unsafe {
            poseidon2_bb31_16_gpu::hash(
                input_device.as_slice().raw(),
                N_INPUT,
                output_device.as_slice().raw(),
                external_rc_device.as_slice().raw(),
                internal_rc_device.as_slice().raw(),
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
