#pragma once

#include <cuda/atomic>

#include "../fields/bb31_t.cuh"

static constexpr const int WIDTH = poseidon2_bb31_16::BabyBear::WIDTH;
static constexpr const int RATE = poseidon2_bb31_16::constants::RATE;

namespace duplex_challenger {
// class DuplexChallenger {
//   public:
//     using F_t = bb31_t;
//     using pF_t = const F_t;

//     F_t sponge_state[WIDTH];
//     F_t input_buffer[RATE];
//     size_t input_buffer_size;
//     F_t output_buffer[WIDTH];
//     size_t output_buffer_size;
// };

__device__ void duplexing(
    bb31_t sponge_state[WIDTH],
    bb31_t input_buffer[RATE],
    bb31_t output_buffer[WIDTH],
    size_t input_buffer_size,
    size_t output_buffer_size
) {
    // Assert input size doesn't exceed RATE
    assert(input_buffer_size <= poseidon2_bb31_16::constants::RATE);

    // Copy input buffer elements to sponge state
    for (size_t i = 0; i < input_buffer_size; i++) {
        sponge_state[i] = input_buffer[i];
    }

    // Clear input buffer
    input_buffer_size = 0;

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

    output_buffer_size = WIDTH;
    for (size_t i = 0; i < WIDTH; i++) {
        sponge_state[i] = output_buffer[i];
    }
}

__device__ void observe(
    bb31_t sponge_state[WIDTH],
    bb31_t input_buffer[RATE],
    bb31_t output_buffer[WIDTH],
    size_t input_buffer_size,
    size_t output_buffer_size,
    bb31_t value
) {
    output_buffer_size = 0;
    input_buffer_size += 1;
    input_buffer[input_buffer_size - 1] = value;

    if (input_buffer_size == poseidon2_bb31_16::constants::RATE) {
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
    bb31_t sponge_state[WIDTH],
    bb31_t input_buffer[RATE],
    bb31_t output_buffer[WIDTH],
    size_t input_buffer_size,
    size_t output_buffer_size
) {
    bb31_t result;
    if (input_buffer_size != 0 || output_buffer_size == 0) {
        duplexing(
            sponge_state,
            input_buffer,
            output_buffer,
            input_buffer_size,
            output_buffer_size
        );
    }
    result = output_buffer[output_buffer_size - 1];
    output_buffer_size -= 1;
    return result;
}

__device__ size_t sample_bits(
    bb31_t sponge_state[WIDTH],
    bb31_t input_buffer[RATE],
    bb31_t output_buffer[WIDTH],
    size_t input_buffer_size,
    size_t output_buffer_size,
    size_t bits
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
    return rand_usize & ((1 << bits) - 1);
}

__device__ bool check_witness(
    bb31_t sponge_state[WIDTH],
    bb31_t input_buffer[RATE],
    bb31_t output_buffer[WIDTH],
    size_t input_buffer_size,
    size_t output_buffer_size,
    size_t bits,
    bb31_t witness
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

    // if (threadIdx.x ==0) {
    //     for (size_t i = 0; i < WIDTH; i++) {
    //         printf("Grinding sponge start at index %d, %d \n ", i, sponge_state[i]);
    //     }
    //     printf("\n");
    // }
    bb31_t sponge_state_clone[WIDTH];
    bb31_t input_buffer_clone[RATE];
    bb31_t output_buffer_clone[WIDTH];

    for (size_t i = idx; i < n || !*found_flag; i += blockDim.x * gridDim.x) {
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
                input_buffer_size,
                output_buffer_size,
                bits,
                witness
            )) {
            out[0] = witness;

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
        bb31_t (* input_buffer)[WIDTH],
        bb31_t (* sponge_state)[WIDTH],
        bb31_t * output_buffer,
        size_t input_buffer_size,
        size_t output_buffer_size,
        size_t bits,
        size_t n,
        bb31_t * out,
        size_t nThreadsPerBlock,
        cudaStream_t stream
    ) {
        printf("Calling grind\n");

        int* d_found_flag;
        cudaMalloc(&d_found_flag, sizeof(int));
        cudaMemset(d_found_flag, 0, sizeof(int));

        size_t nBlocksPerKernel = 512;

        printf("nBlocksPerKernel: %d\n", nBlocksPerKernel);
        printf("bits, n: %d, %d\n", bits, n);

        printf("One: %d\n", 1);
        printf("Coerced to bb31_t: %d\n", bb31_t((int)1));
        printf("Coerced to size_t: %d\n", (size_t)bb31_t((int)1));

        // for (size_t i = 0; i < WIDTH; i++) {
        //     printf("sponge start at index %d, %d \n ", i, sponge_state[0][i]);
        // }

        poseidon2_baby_bear_kernels::permute<<<1, 1>>>(
            sponge_state,
            input_buffer,
            1
        );

        // for (size_t i = 0; i < WIDTH; i++) {
        //     printf(
        //         "after permute, sponge start at index %d, %d \n ",
        //         i,
        //         sponge_state[0][i]
        //     );
        // }

        // duplex_challenger::
        //     grind<<<nBlocksPerKernel, nThreadsPerBlock, 0, stream>>>(
        //         out,
        //         input_buffer,
        //         sponge_state,
        //         output_buffer,
        //         d_found_flag,
        //         input_buffer_size,
        //         output_buffer_size,
        //         bits,
        //         n
        //     );

        // grind(
        //     bb31_t* input_buffer,
        //     bb31_t* sponge_state,
        //     bb31_t* output_buffer,
        //     size_t input_buffer_size,
        //     size_t output_buffer_size,
        //     size_t bits,
        //     size_t n,
        //     bb31_t* out,
        //     int* found_flag
        // )
    }
}  // namespace grinding_challenger_gpu