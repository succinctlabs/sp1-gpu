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

    // Clear input buffer
    *input_buffer_size = 0;

    // Apply the permutation
    poseidon2::BabyBearHasher hasher;
    // if (threadIdx.x == 0) {
    //     for (size_t i = 0; i < WIDTH; i++) {
    //         printf("sponge start at index %d, %d \n ", i, sponge_state[i]);
    //     }
    //     printf("\n");
    // }
    hasher.permute(sponge_state, output_buffer);
    // if (threadIdx.x == 0) {
    //     for (size_t i = 0; i < WIDTH; i++) {
    //         printf("output buffer at index %d, %d \n ", i, output_buffer[i]);
    //     }
    //     printf("\n");
    // }

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
    *output_buffer_size = 0;
    *input_buffer_size += 1;
    input_buffer[*input_buffer_size - 1] = *value;

    // sponge_state[0] = input_buffer[*input_buffer_size - 1];

    // printf("New input buffer size %d\n", input_buffer_size);
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
    // Some assertions.
    bb31_t rand_f = sample(
        sponge_state,
        input_buffer,
        output_buffer,
        input_buffer_size,
        output_buffer_size
    );
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
    // Compute the current value of
    size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    out[0] = bb31_t(56474);

    bb31_t sponge_state_clone[WIDTH];
    bb31_t input_buffer_clone[RATE];
    bb31_t output_buffer_clone[WIDTH];

    for (size_t i = idx; i < n && !*found_flag; i += blockDim.x * gridDim.x) {
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
        // sponge_state[0] = witness;
        // atomicExch(found_flag, 1);
        bool val = check_witness(
            sponge_state_clone,
            input_buffer_clone,
            output_buffer_clone,
            &input_buffer_size,
            &output_buffer_size,
            &bits,
            &witness
        );

        // sponge_state[0] = bb31_t(val);

        // {
        if (val) {
            out[0] = witness;
            atomicExch(found_flag, 1);
            return;
        }
        //     out[0] = bb31_t((int)32583475);
        // }

        //     // Send a message to the other threads that they can terminate.

        // }
    }

    __syncthreads();
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
        // printf("bits, n: %d, %d\n", bits, n);

        int* d_found_flag;
        cudaMalloc(&d_found_flag, sizeof(int));
        cudaMemset(d_found_flag, 0, sizeof(int));

        // size_t* input_buffer_size_ptr;
        // cudaMalloc(&input_buffer_size_ptr,sizeof(size_t));
        // cudaMemcpy(input_buffer_size_ptr, &input_buffer_size, sizeof(size_t), cudaMemcpyHostToDevice);

        // size_t* output_buffer_size_ptr;
        // cudaMalloc(&output_buffer_size_ptr,sizeof(size_t));
        // cudaMemcpy(output_buffer_size_ptr, &output_buffer_size, sizeof(size_t), cudaMemcpyHostToDevice);

        duplex_challenger::grind<<<32, 32>>>(
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