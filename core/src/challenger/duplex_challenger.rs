use p3_baby_bear::BabyBear;
use p3_challenger::GrindingChallenger;
use p3_field::{AbstractField, PrimeField64};
use sp1_recursion_core::stark::OuterChallenger;
use sp1_stark::InnerChallenger;

use crate::{
    cuda_runtime::stream::CudaStreamHandle,
    device::{
        memory::{copy_host_to_device, ToDevice, ToHost},
        DeviceBuffer,
    },
    poseidon2::baby_bear::poseidon2_baby_bear_16_kernels::{RATE, WIDTH},
};

pub trait GrindOnDevice: GrindingChallenger {
    fn grind_on_device(&mut self, bits: usize) -> Self::Witness;
}

impl GrindOnDevice for InnerChallenger {
    fn grind_on_device(&mut self, bits: usize) -> Self::Witness {
        let mut result = vec![BabyBear::zero()];
        println!("Sponge state size: {}", self.sponge_state.len());
        let mut result_d = result.to_device().unwrap();

        let mut sponge_d = vec![self.sponge_state].to_device().unwrap();
        let mut input_d = vec![[BabyBear::zero(); WIDTH]].to_device().unwrap();
        // let input_d = self.input_buffer.to_device().unwrap();
        let output_d = self.output_buffer.to_device().unwrap();
        unsafe {
            result_d.set_len(1);
            input_d.set_len(1);
            grind(
                input_d.as_mut_ptr(),
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

        // self.input_buffer = input_buffer_buffer.to_host();
        // self.sponge_state = sponge_buffer.to_host().try_into().unwrap();
        // self.output_buffer = output_buffer_buffer.to_host();

        // result = result_d.to_host();

        // self.check_witness(bits, result[0]);

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
    pub fn grind(
        input_buffer: *mut [BabyBear; WIDTH],
        sponge_state: *mut [BabyBear; WIDTH],
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
