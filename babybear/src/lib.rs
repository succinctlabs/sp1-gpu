use p3_baby_bear::BabyBear;

pub mod constants;

extern "C" {
    pub fn poseidon2_external_linear_layer_cuda(
        state: *const BabyBear,
        n: usize,
        num_blocks: usize,
        threads_per_block: usize,
    );
    pub fn poseidon2_internal_linear_layer_cuda(
        state: *const BabyBear,
        n: usize,
        num_blocks: usize,
        threads_per_block: usize,
    );
    pub fn poseidon2_permute_cuda(
        input: *const BabyBear,
        output: *mut BabyBear,
        external_rc: *const BabyBear,
        internal_rc: *const BabyBear,
        n: usize,
        num_blocks: usize,
        threads_per_block: usize,
    );
    pub fn poseidon2_hash_cuda(
        input: *const BabyBear,
        len: usize,
        output: *mut BabyBear,
        external_rc: *const BabyBear,
        internal_rc: *const BabyBear,
        n: usize,
        num_blocks: usize,
        threads_per_block: usize,
    );
}

#[cfg(test)]
mod tests {
    use crate::constants::RC_16_30;
    use crate::poseidon2_external_linear_layer_cuda;
    use crate::poseidon2_hash_cuda;
    use crate::poseidon2_internal_linear_layer_cuda;
    use crate::poseidon2_permute_cuda;
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

    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 13;
    const WIDTH: usize = 16;
    const DIGEST_WIDTH: usize = 8;
    const RATE: usize = 8;
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
        (external_round_constants, internal_round_constants)
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
    fn test_external_linear_layer_cuda() {
        let mut rng = thread_rng();
        let n = 100;
        let threads_per_block = 32;
        let num_blocks = n / threads_per_block + 1;
        let input = (0..n * WIDTH)
            .map(|_| rng.gen::<BabyBear>())
            .collect::<Vec<_>>();

        // Execute the source implementation.
        let matrix = Poseidon2ExternalMatrixGeneral;
        let mut results = Vec::new();
        for i in 0..n {
            let state: [BabyBear; WIDTH] = input[i * WIDTH..(i + 1) * WIDTH].try_into().unwrap();
            results.extend(matrix.permute(state).to_vec());
        }

        // Execute the kernel.
        unsafe {
            poseidon2_external_linear_layer_cuda(input.as_ptr(), n, num_blocks, threads_per_block);
        }

        // Check the results.
        for (a, b) in input.iter().zip(results.iter()) {
            assert_eq!(*a, *b);
        }
    }

    #[test]
    fn test_internal_linear_layer_cuda() {
        let mut rng = thread_rng();
        let n = 1;
        let threads_per_block = 32;
        let num_blocks = n / threads_per_block + 1;
        let input = (0..n * WIDTH)
            .map(|_| rng.gen::<BabyBear>())
            .collect::<Vec<_>>();

        // Execute the source implementation.
        let matrix = DiffusionMatrixBabyBear;
        let mut results = Vec::new();
        for i in 0..n {
            let state: [BabyBear; WIDTH] = input[i * WIDTH..(i + 1) * WIDTH].try_into().unwrap();
            results.extend(matrix.permute(state).to_vec());
        }

        // Execute the kernel.
        unsafe {
            poseidon2_internal_linear_layer_cuda(input.as_ptr(), n, num_blocks, threads_per_block);
        }

        // Check the results.
        for (a, b) in input.iter().zip(results.iter()) {
            assert_eq!(*a, *b);
        }
    }

    #[test]
    fn test_permute_cuda() {
        let mut rng = thread_rng();
        let n = 2;
        let threads_per_block = 32;
        let num_blocks = n / threads_per_block + 1;
        let input = (0..n * WIDTH)
            .map(|_| rng.gen::<BabyBear>())
            .collect::<Vec<_>>();

        // Execute the source implementation.
        let poseidon2 = perm();
        let mut results = Vec::new();
        for i in 0..n {
            let state: [BabyBear; WIDTH] = input[i * WIDTH..(i + 1) * WIDTH].try_into().unwrap();
            results.extend(poseidon2.permute(state).to_vec());
        }

        // Execute the kernel.
        let (external_round_constants, internal_round_constants) = round_constants();
        let mut output = Vec::new();
        output.resize(results.len(), BabyBear::zero());
        unsafe {
            poseidon2_permute_cuda(
                input.as_ptr(),
                output.as_mut_ptr(),
                external_round_constants
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .as_ptr(),
                internal_round_constants.clone().as_ptr(),
                n,
                num_blocks,
                threads_per_block,
            );
        }

        for (a, b) in results.iter().zip(output.iter()) {
            assert_eq!(*a, *b);
        }
    }

    #[test]
    fn test_hash_cuda() {
        let mut rng = thread_rng();
        let n = 1;
        let threads_per_block = 32;
        let num_blocks = n / threads_per_block + 1;
        let input = (0..n * WIDTH)
            .map(|_| rng.gen::<BabyBear>())
            .collect::<Vec<_>>();

        // Execute the source implementation.
        let perm = perm();
        let sponge: PaddingFreeSponge<
            Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>,
            WIDTH,
            RATE,
            DIGEST_WIDTH,
        > = PaddingFreeSponge::new(perm);
        let digest = sponge.hash_iter(input.clone());

        // Execute the kernel.
        let (external_round_constants, internal_round_constants) = round_constants();
        let mut output = Vec::new();
        output.resize(DIGEST_WIDTH, BabyBear::zero());
        unsafe {
            poseidon2_hash_cuda(
                input.as_ptr(),
                WIDTH,
                output.as_mut_ptr(),
                external_round_constants
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .as_ptr(),
                internal_round_constants.clone().as_ptr(),
                n,
                num_blocks,
                threads_per_block,
            );
        }

        for (a, b) in digest.iter().zip(output.iter()) {
            assert_eq!(*a, *b);
        }
    }
}
