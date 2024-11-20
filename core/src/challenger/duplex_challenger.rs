use p3_baby_bear::BabyBear;
use p3_challenger::GrindingChallenger;
use p3_field::{AbstractField, PrimeField64};
use sp1_recursion_core::stark::OuterChallenger;
use sp1_stark::InnerChallenger;

use crate::{
    cuda_runtime::stream::CudaStreamHandle,
    device::memory::{ToDevice, ToHost},
    poseidon2::baby_bear::poseidon2_baby_bear_16_kernels::{RATE, WIDTH},
};

pub trait GrindOnDevice: GrindingChallenger {
    fn grind_on_device(&mut self, bits: usize) -> Self::Witness;
}

impl GrindOnDevice for InnerChallenger {
    fn grind_on_device(&mut self, bits: usize) -> Self::Witness {
        // Initialize the result and move it to the device.
        let result = vec![BabyBear::zero()];
        let mut result_d = result.to_device().unwrap();

        // Move the challenger state to device.
        let mut sponge_d = self.sponge_state.to_device().unwrap();

        let input_array: [BabyBear; RATE] = std::array::from_fn(|i| {
            if i < self.input_buffer.len() {
                self.input_buffer[i]
            } else {
                BabyBear::zero()
            }
        });
        let input_d = input_array.to_device().unwrap();

        let output_array: [BabyBear; WIDTH] = std::array::from_fn(|i| {
            if i < self.output_buffer.len() {
                self.output_buffer[i]
            } else {
                BabyBear::zero()
            }
        });
        let output_d = output_array.to_device().unwrap();

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
            );
        }

        let result = result_d.to_host();

        // Check the witness. This is necessary, because it changes the internal state of the
        // challenger, and the CPU version of the challenger does this as well. (It's also necessary
        // for the security of the protocol.)
        let _val = self.check_witness(bits, result[0]);

        debug_assert!(_val);

        result[0]
    }
}

impl GrindOnDevice for OuterChallenger {
    fn grind_on_device(&mut self, bits: usize) -> Self::Witness {
        self.grind(bits)
    }
}

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
    );
}
