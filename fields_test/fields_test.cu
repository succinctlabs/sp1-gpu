#include <iostream>
#include <cuda_runtime.h>
#include "bb31_t.cuh"
#include "bb31_extension_t.cuh"
#include "binary/binary_tower.cuh"
#include "../binius-gpu-main/src/ulvt/finite_fields/circuit_generator/unrolled/binary_tower_unrolled.cuh"
#include "../binius-gpu-main/src/ulvt/finite_fields/circuit_generator/unrolled/binary_tower_unrolled5.cu"
#include "../binius-gpu-main/src/ulvt/finite_fields/circuit_generator/unrolled/binary_tower_unrolled7.cu"

#include "../binius-gpu-main/src/ulvt/utils/bitslicing.cuh"

// This kernel performs a specified number of uint32_t multiplications per thread.
// Each iteration does: x = a * x, which can be considered 1 operation.
template< typename E, typename T, int NUM_ITERATIONS>
__global__ void mulKernel(E *out, T a, E b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    E x = b;

    #pragma unroll
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        x = a * x; // 1 operation per iteration (1 mul)
      asm volatile("");
    }

    // Write out the final value to ensure the compiler doesn't optimize away the computation
    out[idx] = x;
}

template<typename E, typename T, int NUM_ITERATIONS>
__global__ void addKernel(E *out, T a, E b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    E x = b;

    #pragma unroll
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        x = a + x; // 1 operation per iteration (1 add)
        asm volatile("");
    }

    // Write out the final value to ensure the compiler doesn't optimize away the computation
    out[idx] = x;
}

template<int NUM_ITERATIONS>
__global__ void binaryBaseExtMultiplication(__uint128_t *out, uint32_t a, __uint128_t b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __uint128_t x = b;

    #pragma unroll 
    for (int i = 0; i < NUM_ITERATIONS; i++) {

        __uint128_t mask = -(__uint128_t)a;
        x = x & mask; // 1 operation per iteration (1 add)
        asm volatile("");
    }

    // Write out the final value to ensure the compiler doesn't optimize away the computation
    out[idx] = x;
}

template<int NUM_ITERATIONS, int POWER_HEIGHT, int NUM_WORDS>
__global__ void binaryMultiplicationBitSliced(uint32_t *out, uint32_t *a, uint32_t *b) {
    const int WIDTH = 1 << POWER_HEIGHT;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x[WIDTH];
    for (int i = 0; i < NUM_WORDS; i++) {
        x[i] = b[i];
    }

    uint32_t y[WIDTH];
    for (int i = 0; i < NUM_WORDS; i++) {
        y[i] = a[i];
    }

    #pragma unroll 1 
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // BitsliceUtils<WIDTH>::bitslice_transpose(y);
	    // BitsliceUtils<WIDTH>::bitslice_transpose(x);

        multiply_unrolled<POWER_HEIGHT>(y, x, x);

        // BitsliceUtils<WIDTH>::bitslice_untranspose(x);

        asm volatile("");
    }

    for (int i = 0; i < NUM_WORDS; i++) {
        out[idx * NUM_WORDS + i] = x[i];
    }
}



template<typename E, typename T, int NUM_ITERATIONS>
void run_benchmark(void (*opKernel)(E *out, T a, E b), T a, E b) {
        // GPU parameters
    int threadsPerBlock = 256;
    int numBlocks = 8192;  // Adjust to fully load your GPU
    int totalThreads = threadsPerBlock * numBlocks;

    // Host and device pointers
    E *d_out;
    cudaMalloc((void**)&d_out, totalThreads * sizeof(E));
    // cudaMemset(d_out, 0, totalThreads * sizeof(T));

    // Use CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up launch (optional) to remove first-time overheads
    opKernel<<<numBlocks, threadsPerBlock>>>(d_out, a, b);
    cudaDeviceSynchronize();

    // Start timing
    cudaEventRecord(start);

    // Actual benchmark kernel launch
    opKernel<<<numBlocks, threadsPerBlock>>>(d_out, a, b);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Compute operations
    double opsPerThread = (double)NUM_ITERATIONS; // 1 operation per iteration
    double totalOps = opsPerThread * (double)totalThreads;
    double seconds = milliseconds / 1000.0;

    double ops = totalOps / seconds;

    std::cout << "Total operations: " << totalOps << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "Operations per second: " << ops << "\n";
    std::cout << "TOP/s: " << (ops / 1e12) << "\n";

    // Clean up
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

template<int NUM_ITERATIONS, int POWER_HEIGHT, int NUM_WORDS>
void run_benchmark_binary_bitsliced() {
    static_assert(NUM_WORDS <= (1 << POWER_HEIGHT));
    const int NUM_ELEMENTS =  NUM_WORDS;
        // GPU parameters
    int threadsPerBlock = 256;
    int numBlocks = 4;  // Adjust to fully load your GPU
    int totalThreads = threadsPerBlock * numBlocks;

    // Host and device pointers
    uint32_t *d_out;
    cudaMalloc((void**)&d_out, totalThreads * sizeof(uint32_t) * NUM_WORDS);

    // Input pointers
    uint32_t *d_a;
    uint32_t *d_b;
    cudaMalloc((void**)&d_a, sizeof(uint32_t) * NUM_WORDS);
    cudaMalloc((void**)&d_b, sizeof(uint32_t) * NUM_WORDS);

    // Allocate and initialize host arrays
    uint32_t *h_a = new uint32_t[NUM_WORDS];
    uint32_t *h_b = new uint32_t[NUM_WORDS];
    
    // Fill host arrays with random values
    for (int i = 0; i < NUM_WORDS; i++) {
        h_a[i] = rand();
        h_b[i] = rand();
    }

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, sizeof(uint32_t) * NUM_WORDS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(uint32_t) * NUM_WORDS, cudaMemcpyHostToDevice);


    // Use CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up launch (optional) to remove first-time overheads
    binaryMultiplicationBitSliced<NUM_ITERATIONS, POWER_HEIGHT, NUM_WORDS><<<numBlocks, threadsPerBlock>>>(d_out, d_a, d_b);
    cudaDeviceSynchronize();

    // Start timing
    cudaEventRecord(start);

    // Actual benchmark kernel launch
    binaryMultiplicationBitSliced<NUM_ITERATIONS, POWER_HEIGHT, NUM_WORDS><<<numBlocks, threadsPerBlock>>>(d_out, d_a, d_b);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Compute operations
    double opsPerThread = (double)NUM_ITERATIONS * (double)NUM_ELEMENTS; 
    double totalOps = opsPerThread * (double)totalThreads;
    double seconds = milliseconds / 1000.0;

    double ops = totalOps / seconds;

    std::cout << "Total operations: " << totalOps << "\n";
    std::cout << "Time (s): " << seconds << "\n";
    std::cout << "Operations per second: " << ops << "\n";
    std::cout << "TOP/s: " << (ops / 1e12) << "\n";

    // Clean up
    cudaFree(d_out);
    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

int main() {

    const int NUM_ITERATIONS = 1000000;

    // Get a baseiine by testing the number of FLOP multiplications
    std::cout << "Testing floating point multiplications..." << std::endl;
    float a_fp32 = 2;
    float b_fp32 = 1;
    run_benchmark<float, float, NUM_ITERATIONS>(mulKernel<float, float, NUM_ITERATIONS>, a_fp32, b_fp32);
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Testing integer additions..." << std::endl;
    uint32_t a_uint32_add = rand();
    uint32_t b_uint32_add = rand();
    run_benchmark<uint32_t, uint32_t, NUM_ITERATIONS>(addKernel<uint32_t, uint32_t, NUM_ITERATIONS>, a_uint32_add, b_uint32_add);
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Testing integer multiplications..." << std::endl;
    uint32_t a_uint32_mul = rand();
    uint32_t b_uint32_mul = rand();
    run_benchmark<uint32_t, uint32_t, NUM_ITERATIONS>(mulKernel<uint32_t, uint32_t, NUM_ITERATIONS>, a_uint32_mul, b_uint32_mul);
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Testing 64-bit integer multiplications..." << std::endl;
    uint64_t a_uint64 = rand();
    uint64_t b_uint64 = rand();
    run_benchmark<uint64_t, uint64_t, NUM_ITERATIONS>(mulKernel<uint64_t, uint64_t, NUM_ITERATIONS>, a_uint64, b_uint64);
    std::cout << "----------------------------------------" << std::endl;
    
    std::cout << "Testing BB31 additions..." << std::endl;
    bb31_t a_bb31_add = bb31_t((int)rand());
    bb31_t b_bb31_add = bb31_t((int)rand());
    run_benchmark<bb31_t, bb31_t, NUM_ITERATIONS>(addKernel<bb31_t, bb31_t, NUM_ITERATIONS>, a_bb31_add, b_bb31_add);
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Testing BB31 multiplications..." << std::endl;
    bb31_t a_bb31_mul = bb31_t((int)rand());
    bb31_t b_bb31_mul = bb31_t((int)rand());
    run_benchmark<bb31_t, bb31_t, NUM_ITERATIONS>(mulKernel<bb31_t, bb31_t, NUM_ITERATIONS>, a_bb31_mul, b_bb31_mul);
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Testing BB31 extension additions..." << std::endl;
    bb31_t values[4] = {bb31_t((int)rand()), bb31_t((int)rand()), bb31_t((int)rand()), bb31_t((int)rand())};
    bb31_extension_t a_bb31_extension_add = bb31_extension_t(values);
    bb31_t values2[4] = {bb31_t((int)rand()), bb31_t((int)rand()), bb31_t((int)rand()), bb31_t((int)rand())};
    bb31_extension_t b_bb31_extension_add = bb31_extension_t(values2);
    run_benchmark<bb31_extension_t, bb31_extension_t, NUM_ITERATIONS>(addKernel<bb31_extension_t, bb31_extension_t, NUM_ITERATIONS>, a_bb31_extension_add, b_bb31_extension_add);
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Testing BB31 extension multiplications..." << std::endl;
    bb31_t values3[4] = {bb31_t((int)rand()), bb31_t((int)rand()), bb31_t((int)rand()), bb31_t((int)rand())};
    bb31_extension_t a_bb31_extension_mul = bb31_extension_t(values3);
    bb31_t values4[4] = {bb31_t((int)rand()), bb31_t((int)rand()), bb31_t((int)rand()), bb31_t((int)rand())};
    bb31_extension_t b_bb31_extension_mul = bb31_extension_t(values4);
    run_benchmark<bb31_extension_t, bb31_extension_t, NUM_ITERATIONS>(mulKernel<bb31_extension_t, bb31_extension_t, NUM_ITERATIONS>, a_bb31_extension_mul, b_bb31_extension_mul);
    std::cout << "----------------------------------------" << std::endl;


    std::cout << "Testing BB31 base-extension multiplications..." << std::endl;
    bb31_t a_bb31_base_mul = bb31_t((int)rand());
    bb31_extension_t b_bb31_base_mul = bb31_extension_t(values4);
    run_benchmark<bb31_extension_t, bb31_t, NUM_ITERATIONS>(mulKernel<bb31_extension_t, bb31_t, NUM_ITERATIONS>, a_bb31_base_mul, b_bb31_base_mul);
    std::cout << "----------------------------------------" << std::endl;


    std::cout << "Testing f2_t<5> additions..." << std::endl;
    f2_t<5> a_fp25_add = f2_t<5>::from_inner(rand());
    f2_t<5> b_fp25_add = f2_t<5>::from_inner(rand());
    run_benchmark<f2_t<5>, f2_t<5>, NUM_ITERATIONS>(addKernel<f2_t<5>, f2_t<5>, NUM_ITERATIONS>, a_fp25_add, b_fp25_add);
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Testing f2_t<5> multiplications..." << std::endl;
    f2_t<5> a_fp25_mul = f2_t<5>::from_inner(rand());
    f2_t<5> b_fp25_mul = f2_t<5>::from_inner(rand());
    run_benchmark<f2_t<5>, f2_t<5>, NUM_ITERATIONS>(mulKernel<f2_t<5>, f2_t<5>, NUM_ITERATIONS>, a_fp25_mul, b_fp25_mul);
    std::cout << "----------------------------------------" << std::endl;


    std::cout << "Testing f2_t<0> multiplications..." << std::endl;
    f2_t<0> a_fp20_mul = f2_t<0>::from_inner(rand());
    f2_t<0> b_fp20_mul = f2_t<0>::from_inner(rand());
    run_benchmark<f2_t<0>, f2_t<0>, NUM_ITERATIONS>(mulKernel<f2_t<0>, f2_t<0>, NUM_ITERATIONS>, a_fp20_mul, b_fp20_mul);
    std::cout << "----------------------------------------" << std::endl;


    std::cout << "Testing binary base-extension multiplications..." << std::endl;
    uint32_t a_binary_mul = rand();
    __uint128_t b_binary_mul = rand();
    run_benchmark<__uint128_t, uint32_t, NUM_ITERATIONS>(binaryBaseExtMultiplication<NUM_ITERATIONS>, a_binary_mul, b_binary_mul);
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Testing binary bitsliced multiplications for 32-bit fields..." << std::endl;
    run_benchmark_binary_bitsliced<NUM_ITERATIONS, 5, 6>();
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Testing binary bitsliced multiplications for 128-bit fields..." << std::endl;
    run_benchmark_binary_bitsliced<NUM_ITERATIONS, 7, 16>();
    std::cout << "----------------------------------------" << std::endl;

    return 0;
}
