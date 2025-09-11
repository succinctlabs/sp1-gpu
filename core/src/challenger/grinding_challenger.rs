use p3_baby_bear::BabyBear;
use p3_challenger::GrindingChallenger;
use p3_field::{AbstractField, PrimeField64};
use sp1_recursion_core::stark::OuterChallenger;
use sp1_stark::InnerChallenger;

use crate::{
    cuda_runtime::stream::{CudaStream, CudaStreamHandle},
    device::{
        error::CudaRustError,
        memory::{ToDevice, ToHost},
    },
    poseidon2::baby_bear::poseidon2_baby_bear_16_kernels::{RATE, WIDTH},
};

/// A [`GrindingChallenger`] that can also grind on device.
///
/// Useful for finding a proof-of-work witness on machines with not that many cores.
pub trait DeviceGrindingChallenger: GrindingChallenger {
    /// Grinds on device.
    fn grind_device(&mut self, bits: usize, stream: &CudaStream) -> Self::Witness;
}

lazy_static::lazy_static! {
    static ref DISABLE_GRIND_DEVICE: bool = std::env::var("MOONGATE_DISABLE_GRIND_DEVICE")
        .unwrap_or("false".to_string())
        .parse::<bool>()
        .unwrap_or(false);
}

impl DeviceGrindingChallenger for InnerChallenger {
    fn grind_device(&mut self, bits: usize, stream: &CudaStream) -> Self::Witness {
        if *DISABLE_GRIND_DEVICE {
            return self.grind(bits);
        }

        // Initialize the result and move it to the device.
        let result = vec![BabyBear::zero()];
        let mut result_d = result.to_device_async(stream).unwrap();

        // Move the challenger state to device.
        let mut sponge_d = self.sponge_state.to_device_async(stream).unwrap();
        let input_array: [BabyBear; RATE] = std::array::from_fn(|i| {
            if i < self.input_buffer.len() {
                self.input_buffer[i]
            } else {
                BabyBear::zero()
            }
        });
        let input_d = input_array.to_device_async(stream).unwrap();
        let output_array: [BabyBear; WIDTH] = std::array::from_fn(|i| {
            if i < self.output_buffer.len() {
                self.output_buffer[i]
            } else {
                BabyBear::zero()
            }
        });
        let output_d = output_array.to_device_async(stream).unwrap();

        // Grind on device.
        unsafe {
            result_d.set_len(1);
            grind_baby_bear(
                input_d.as_ptr(),
                sponge_d.as_mut_ptr(),
                output_d.as_ptr(),
                self.input_buffer.len(),
                self.output_buffer.len(),
                bits,
                BabyBear::ORDER_U64 as usize,
                result_d.as_mut_ptr(),
                512,
                input_d.stream().handle(),
            )
            .to_result()
            .unwrap();
        }

        let result = result_d.to_host();

        // Check the witness. This is necessary, because it changes the internal state of the
        // challenger, and the CPU version of the challenger does this as well. It's also necessary
        // for the security of the protocol.
        assert!(self.check_witness(bits, result[0]));

        result[0]
    }
}

/// The implementation for the OuterChallenger is identical to its underlying host implementation.
impl DeviceGrindingChallenger for OuterChallenger {
    fn grind_device(&mut self, bits: usize, _stream: &CudaStream) -> Self::Witness {
        self.grind(bits)
    }
}

/// cbindgen:ignore
#[allow(unused_attributes)]
#[link_name = "duplex_challenger"]
extern "C" {
    pub fn grind_baby_bear(
        input_buffer: *const BabyBear,
        sponge_state: *mut BabyBear,
        output_buffer: *const BabyBear,
        input_buffer_size: usize,
        output_buffer_size: usize,
        bits: usize,
        n: usize,
        out: *mut BabyBear,
        nThreadsPerBlock: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
    use p3_field::AbstractField;
    use sp1_stark::{inner_perm, InnerChallenger};

    use crate::{
        challenger::grinding_challenger::DeviceGrindingChallenger, cuda_runtime::stream::CudaStream,
    };

    #[test]
    fn test_grinding() {
        (1..20).for_each(|bits| {
            let mut challenger = InnerChallenger::new(inner_perm());

            let stream = CudaStream::default();

            // Observe 7 elements to make the input buffer almost full and trigger duplexing on
            // device.
            challenger.observe(BabyBear::from_canonical_u32(0xDEADBEEF));
            challenger.observe(BabyBear::from_canonical_u32(0xCAFEBABE));
            challenger.observe(BabyBear::from_canonical_u32(0xDEADBEEF));
            challenger.observe(BabyBear::from_canonical_u32(0xCAFEBABE));
            challenger.observe(BabyBear::from_canonical_u32(0xDEADBEEF));
            challenger.observe(BabyBear::from_canonical_u32(0xCAFEBABE));
            challenger.observe(BabyBear::from_canonical_u32(0xDEADBEEF));

            // Make another challenger that also samples before grinding (this empties the input buffer).
            let mut challenger_2 = challenger.clone();
            let _: BabyBear = challenger.sample();

            // Clone the original challenger because after grinding on device the internal state
            // of `challenger` will change.
            let mut original_challenger = challenger.clone();
            let result = challenger.grind_device(bits, &stream);

            assert!(original_challenger.check_witness(bits, result));

            let mut original_challenger_2 = challenger_2.clone();
            let result_2 = challenger_2.grind_device(bits, &stream);

            assert!(original_challenger_2.check_witness(bits, result_2));

            // Checks to make sure the pow witness was properly observed in `grind_on_device`.
            assert!(original_challenger_2.sponge_state == challenger_2.sponge_state);
            assert!(original_challenger_2.input_buffer == challenger_2.input_buffer);
            assert!(original_challenger_2.output_buffer == challenger_2.output_buffer);
        })
    }
}
