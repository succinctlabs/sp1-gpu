#pragma once

#include <cuda/atomic>

#include "../fields/bb31_t.cuh"

namespace duplex_challenger {
class DuplexChallenger {
  public:
    using F_t = bb31_t;
    using pF_t = const F_t;

    static constexpr const int WIDTH = poseidon2_bb31_16::BabyBear::WIDTH;

    F_t sponge_state[WIDTH];
    F_t* input_buffer;
    size_t input_buffer_size;
    F_t* output_buffer;
    size_t output_buffer_size;
};

static constexpr const int WIDTH = poseidon2_bb31_16::BabyBear::WIDTH;

__device__ void duplexing(DuplexChallenger challenger) {
    // Assert input size doesn't exceed RATE
    assert(challenger.input_buffer_size <= poseidon2_bb31_16::constants::RATE);

    // Copy input buffer elements to sponge state
    for (size_t i = 0; i < challenger.input_buffer_size; i++) {
        challenger.sponge_state[i] = challenger.input_buffer[i];
    }

    // Clear input buffer
    challenger.input_buffer_size = 0;

    // Apply the permutation
    poseidon2::BabyBearHasher hasher;
    hasher.permute(challenger.sponge_state, challenger.output_buffer);
}

__device__ void observe(DuplexChallenger challenger, bb31_t value) {
    challenger.output_buffer_size = 0;
    challenger.input_buffer_size += 1;
    challenger.input_buffer[challenger.input_buffer_size - 1] = value;

    if (challenger.input_buffer_size == poseidon2_bb31_16::constants::RATE) {
        duplexing(challenger);
    }
}

__device__ void observe(DuplexChallenger challenger, bb31_t* values, size_t n) {
    for (size_t i = 0; i < n; i++) {
        observe(challenger, values[i]);
    }
}

__device__ bb31_t sample(DuplexChallenger challenger) {
    bb31_t result;
    if (challenger.input_buffer_size != 0
        || challenger.output_buffer_size == 0) {
        duplexing(challenger);
    }
    result = challenger.output_buffer[challenger.output_buffer_size - 1];
    challenger.output_buffer_size -= 1;
    return result;
}

__device__ size_t sample_bits(DuplexChallenger challenger, size_t bits) {
    // Some assertions.
    bb31_t rand_f = sample(challenger);
    size_t rand_usize = (size_t)rand_f;
    return rand_usize & ((1 << bits) - 1);
}

__device__ bool
check_witness(DuplexChallenger challenger, size_t bits, bb31_t witness) {
    observe(challenger, witness);
    return sample_bits(challenger, bits) == 0;
}

__global__ void grind(
    DuplexChallenger challenger,
    size_t bits,
    size_t n,
    bb31_t* out,
    int* found_flag
) {
    // Compute the current value of
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    for (size_t i = idx; i < n && !*found_flag; i += blockDim.x * gridDim.x) {
        DuplexChallenger challenger_clone;

        size_t cloned_input_buffer_size = challenger.input_buffer_size;
        memcpy(
            challenger_clone.input_buffer,
            challenger.input_buffer,
            sizeof(bb31_t) * cloned_input_buffer_size
        );

        size_t cloned_output_buffer_size = challenger.output_buffer_size;
        memcpy(
            challenger_clone.output_buffer,
            challenger.output_buffer,
            sizeof(bb31_t) * cloned_output_buffer_size
        );

        memcpy(
            challenger_clone.sponge_state,
            challenger.sponge_state,
            sizeof(bb31_t) * poseidon2_bb31_16::BabyBear::WIDTH
        );

        challenger_clone.input_buffer_size = cloned_input_buffer_size;
        challenger_clone.output_buffer_size = cloned_output_buffer_size;

        bb31_t witness = (uint32_t)i;
        if (check_witness(challenger_clone, bits, witness)) {
            out[0] = witness;

            // Need to modify the original challenger.
            check_witness(challenger, bits, witness);

            // Send a message to the other threads that they can terminate.
            atomicExch(found_flag, 1);
            return;
        }
    }
}
}  // namespace duplex_challenger

extern "C" namespace grinding_challenger_gpu {
    using namespace duplex_challenger;

    extern "C" void grind(
        bb31_t * input_buffer,
        bb31_t * sponge_state[WIDTH],
        bb31_t * output_buffer,
        size_t input_buffer_size,
        size_t output_buffer_size,
        size_t bits,
        size_t n,
        bb31_t * out,
        size_t nThreadsPerBlock,
        cudaStream_t stream
    ) {
        DuplexChallenger challenger;
        challenger.input_buffer = input_buffer;
        challenger.input_buffer_size = input_buffer_size;
        challenger.output_buffer = output_buffer;
        challenger.output_buffer_size = output_buffer_size;

        int* d_found_flag;
        cudaMalloc(&d_found_flag, sizeof(int));
        cudaMemset(d_found_flag, 0, sizeof(int));
        cudaMemcpy(
            challenger.sponge_state,
            sponge_state,
            sizeof(bb31_t) * poseidon2_bb31_16::BabyBear::WIDTH,
            cudaMemcpyDeviceToDevice
        );

        size_t nBlocksPerKernel =
            (1 << 20 + nThreadsPerBlock - 1) / nThreadsPerBlock;

        duplex_challenger::
            grind<<<nBlocksPerKernel, nThreadsPerBlock, 0, stream>>>(
                challenger,
                bits,
                n,
                out,
                d_found_flag
            );
    }
}  // namespace grinding_challenger_gpu