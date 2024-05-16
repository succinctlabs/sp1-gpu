pub mod constants;

pub mod poseidon2_gpu {
    use p3_baby_bear::BabyBear;

    use crate::device::slice::DeviceSliceRaw;

    #[allow(unused_attributes)]
    #[link_name = "poseidon2_gpu"]
    extern "C" {
        pub fn permute(
            input: DeviceSliceRaw<[BabyBear; 16]>,
            output: DeviceSliceRaw<[BabyBear; 16]>,
            external_rc: DeviceSliceRaw<[BabyBear; 16]>,
            internal_rc: DeviceSliceRaw<BabyBear>,
            n: usize,
            n_blocks: usize,
            n_threads_per_block: usize,
        );
    }
}

#[cfg(test)]
mod tests {

    use crate::device::buffer::DeviceBuffer;
    use crate::poseidon2::constants::RC_16_30;
    use p3_baby_bear::BabyBear;
    use p3_baby_bear::DiffusionMatrixBabyBear;
    use p3_field::AbstractField;
    use p3_poseidon2::Poseidon2;
    use p3_poseidon2::Poseidon2ExternalMatrixGeneral;
    use p3_symmetric::Permutation;
    use rand::thread_rng;
    use rand::Rng;

    use super::poseidon2_gpu;

    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 13;
    const WIDTH: usize = 16;
    const D_U64: u64 = 7;

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
    fn test_permute_cuda() {
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
        let mut input_device = DeviceBuffer::new(n);
        input_device.extend_from_host_slice(&input);
        let mut output_device = DeviceBuffer::new(n);
        output_device.extend_from_host_slice(&output);
        let mut external_rc_device = DeviceBuffer::new(ROUNDS_F);
        external_rc_device.extend_from_host_slice(&external_round_constants);
        let mut internal_rc_device = DeviceBuffer::new(WIDTH * ROUNDS_P);
        internal_rc_device.extend_from_host_slice(&internal_round_constants);

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
            poseidon2_gpu::permute(
                input_device.as_slice().raw(),
                output_device.as_slice_mut().raw(),
                external_rc_device.as_slice().raw(),
                internal_rc_device.as_slice().raw(),
                n,
                num_blocks,
                threads_per_block,
            );
        }

        // Copy the result of the kernel to the host.
        output_device.copy_into_host(&mut output);
        for (a, b) in gt.iter().zip(output.iter()) {
            assert_eq!(*a, *b);
        }
    }
}
