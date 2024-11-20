pub mod duplex_challenger;

pub mod tests {
    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
    use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
    use p3_field::AbstractField;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use sp1_stark::{inner_perm, InnerChallenger};

    use crate::{
        challenger::duplex_challenger::GrindOnDevice,
        cuda_runtime::stream::CudaStream,
        poseidon2::{
            baby_bear::poseidon2_baby_bear_16_kernels::{D_U64, ROUNDS_F, ROUNDS_P, WIDTH},
            constants::RC_16_30,
        },
    };

    fn round_constants() -> (Vec<[BabyBear; 16]>, Vec<BabyBear>) {
        let mut round_constants = RC_16_30.to_vec();
        let internal_start = ROUNDS_F / 2;
        let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
        let internal_round_constants = round_constants
            .drain(internal_start..internal_end)
            .map(|vec| vec[0])
            .collect::<Vec<_>>();
        let external_round_constants = round_constants;
        (external_round_constants[0..ROUNDS_F].to_vec(), internal_round_constants)
    }

    pub fn poseidon2_baby_bear_16_perm(
    ) -> Poseidon2<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7> {
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
    fn test_grinding() {
        let mut challenger = InnerChallenger::new(poseidon2_baby_bear_16_perm());
        println!("Challenger sponge state: {:?}", challenger.sponge_state);
        challenger.observe(BabyBear::from_canonical_u32(0xDEADBEEF));
        challenger.observe(BabyBear::from_canonical_u32(0xCAFEBABE));
        let _elt: BabyBear = challenger.sample();
        println!("Challenger sponge state: {:?}", challenger.sponge_state);
        println!(
            "Challenger input buffer size: {}, output buffer size: {}, sponge state size: {}",
            challenger.input_buffer.len(),
            challenger.output_buffer.len(),
            challenger.sponge_state.len()
        );

        // Clone the original challenger because after grinding the internal state will change.
        let mut original_challenger = challenger.clone();
        let result = challenger.grind_on_device(3);

        println!("Result: {:?}", result);

        // assert!(original_challenger.check_witness(3, result));
    }
}
