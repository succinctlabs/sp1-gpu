#pragma once

#include <cuda/atomic>

#include "../fields/bb31_t.cuh"

static constexpr const int WIDTH = poseidon2_bb31_16::BabyBear::WIDTH;
static constexpr const int RATE = poseidon2_bb31_16::constants::RATE;

namespace duplex_challenger {

__device__ void duplexing(
    bb31_t* sponge_state,
    bb31_t* input_buffer,
    bb31_t* output_buffer,
    size_t* input_buffer_size,
    size_t* output_buffer_size
) {
    // Assert input size doesn't exceed RATE
    assert(*input_buffer_size <= poseidon2_bb31_16::constants::RATE);

    // Copy input buffer elements to sponge state
    for (size_t i = 0; i < *input_buffer_size; i++) {
        sponge_state[i] = input_buffer[i];
    }

    // Clear input buffer.
    *input_buffer_size = 0;

    // Apply the permutation to the sponge state and store the output in the output buffer.
    poseidon2::BabyBearHasher hasher;
    hasher.permute(sponge_state, output_buffer);


    // Copy the output buffer to the sponge state.
    *output_buffer_size = WIDTH;
    for (size_t i = 0; i < WIDTH; i++) {
        sponge_state[i] = output_buffer[i];
    }
}

__device__ void observe(
    bb31_t* sponge_state,
    bb31_t* input_buffer,
    bb31_t* output_buffer,
    size_t* input_buffer_size,
    size_t* output_buffer_size,
    bb31_t* value
) {
    // Clear the output buffer.
    *output_buffer_size = 0;

    // Push value to the input buffer.
    *input_buffer_size += 1;
    input_buffer[*input_buffer_size - 1] = *value;

    if (*input_buffer_size == poseidon2_bb31_16::constants::RATE) {
        duplexing(
            sponge_state,
            input_buffer,
            output_buffer,
            input_buffer_size,
            output_buffer_size
        );
    }
}

__device__ bb31_t sample(
    bb31_t* sponge_state,
    bb31_t* input_buffer,
    bb31_t* output_buffer,
    size_t* input_buffer_size,
    size_t* output_buffer_size
) {
    bb31_t result;
    if (*input_buffer_size != 0 || *output_buffer_size == 0) {
        duplexing(
            sponge_state,
            input_buffer,
            output_buffer,
            input_buffer_size,
            output_buffer_size
        );
    }
    // Pop the last element of the buffer.
    result = output_buffer[*output_buffer_size - 1];
    *output_buffer_size -= 1;
    return result;
}

__device__ size_t sample_bits(
    bb31_t* sponge_state,
    bb31_t* input_buffer,
    bb31_t* output_buffer,
    size_t* input_buffer_size,
    size_t* output_buffer_size,
    size_t* bits
) {
    bb31_t rand_f = sample(
        sponge_state,
        input_buffer,
        output_buffer,
        input_buffer_size,
        output_buffer_size
    );

    // Equivalent to "as_canonical_u32" in the Rust implementation.
    size_t rand_usize = (uint32_t)rand_f;
    return rand_usize & ((1 << *bits) - 1);
}

__device__ bool check_witness(
    bb31_t* sponge_state,
    bb31_t* input_buffer,
    bb31_t* output_buffer,
    size_t* input_buffer_size,
    size_t* output_buffer_size,
    size_t* bits,
    bb31_t* witness
) {
    observe(
        sponge_state,
        input_buffer,
        output_buffer,
        input_buffer_size,
        output_buffer_size,
        witness
    );
    return sample_bits(
               sponge_state,
               input_buffer,
               output_buffer,
               input_buffer_size,
               output_buffer_size,
               bits
           )
        == 0;
}

__global__ void grind(
    bb31_t* out,
    bb31_t* input_buffer,
    bb31_t* sponge_state,
    bb31_t* output_buffer,
    int* found_flag,
    size_t input_buffer_size,
    size_t output_buffer_size,
    size_t bits,
    size_t n
) {
    static constexpr const int WIDTH = poseidon2_bb31_16::BabyBear::WIDTH;
    static constexpr const int RATE = poseidon2_bb31_16::constants::RATE;

    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    size_t original_input_buffer_size = input_buffer_size;
    size_t original_output_buffer_size = output_buffer_size;

    bb31_t sponge_state_clone[WIDTH];
    bb31_t input_buffer_clone[RATE];
    bb31_t output_buffer_clone[WIDTH];

    for (size_t i = idx; i < n && !*found_flag; i += blockDim.x * gridDim.x) {
        // Reset the buffer sizes to their values at the start of the loop (as they were when the 
        // function was called), and make a deep clone of the challenger.
        input_buffer_size = original_input_buffer_size;
        output_buffer_size = original_output_buffer_size;
        for (size_t j = 0; j < input_buffer_size; j++) {
            input_buffer_clone[j] = input_buffer[j];
        }
        for (size_t j = 0; j < output_buffer_size; j++) {
            output_buffer_clone[j] = output_buffer[j];
        }
        for (size_t j = 0; j < poseidon2_bb31_16::BabyBear::WIDTH; j++) {
            sponge_state_clone[j] = sponge_state[j];
        }

        bb31_t witness = bb31_t((int)i);

        if (check_witness(
            sponge_state_clone,
            input_buffer_clone,
            output_buffer_clone,
            &input_buffer_size,
            &output_buffer_size,
            &bits,
            &witness
        )) {
            out[0] = witness;

            // Set the flag to 1 so that other threads can stop.
            atomicExch(found_flag, 1);
            return;
        }
    }
}
}  // namespace duplex_challenger

extern "C" namespace grinding_challenger_gpu {
    using namespace duplex_challenger;

    extern "C" void grind_baby_bear(
        bb31_t * input_buffer,
        bb31_t * sponge_state,
        bb31_t * output_buffer,
        size_t input_buffer_size,
        size_t output_buffer_size,
        size_t bits,
        size_t n,
        bb31_t * out,
        size_t nThreadsPerBlock,
        cudaStream_t stream
    ) {
        // Allocate an atomic flag to signal when a solution is found.
        int* d_found_flag;
        cudaMalloc(&d_found_flag, sizeof(int));
        cudaMemset(d_found_flag, 0, sizeof(int));

        duplex_challenger::grind<<<1, nThreadsPerBlock>>>(
            out,
            input_buffer,
            sponge_state,
            output_buffer,
            d_found_flag,
            input_buffer_size,
            output_buffer_size,
            bits,
            n
        );
    }
}  // namespace grinding_challenger_gpu