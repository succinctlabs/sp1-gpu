#include <iostream>
#include <cuda_runtime.h>
#include "bb31_t.cuh"
// This kernel performs a specified number of uint32_t multiplications per thread.
// Each iteration does: x = a * x, which can be considered 1 operation.
template<typename T, int NUM_ITERATIONS>
__global__ void mulKernel(T *out, T a, T b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T x = b;

    #pragma unroll
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        x = a * x; // 1 operation per iteration (1 mul)
    }

    // Write out the final value to ensure the compiler doesn't optimize away the computation
    out[idx] = x;
}

template<typename T, int NUM_ITERATIONS>
__global__ void addKernel(T *out, T a, T b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T x = b;

    #pragma unroll
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        x = a + x; // 1 operation per iteration (1 add)
    }

    // Write out the final value to ensure the compiler doesn't optimize away the computation
    out[idx] = x;
}


template<typename T, int NUM_ITERATIONS>
void run_benchmark(void (*opKernel)(T *out, T a, T b), T a, T b) {
        // GPU parameters
    int threadsPerBlock = 256;
    int numBlocks = 8192;  // Adjust to fully load your GPU
    int totalThreads = threadsPerBlock * numBlocks;

    // Host and device pointers
    T *d_out;
    cudaMalloc((void**)&d_out, totalThreads * sizeof(T));
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

int main() {

    const int NUM_ITERATIONS = 1000000;

    // Get a baseiine by testing the number of FLOP multiplications
    std::cout << "Testing floating point multiplications..." << std::endl;
    float a_fp32 = 2;
    float b_fp32 = 1;
    run_benchmark<float, NUM_ITERATIONS>(mulKernel<float, NUM_ITERATIONS>, a_fp32, b_fp32);
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Testing integer additions..." << std::endl;
    uint32_t a_uint32_add = rand();
    uint32_t b_uint32_add = rand();
    run_benchmark<uint32_t, NUM_ITERATIONS>(addKernel<uint32_t, NUM_ITERATIONS>, a_uint32_add, b_uint32_add);
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Testing integer multiplications..." << std::endl;
    uint32_t a_uint32_mul = rand();
    uint32_t b_uint32_mul = rand();
    run_benchmark<uint32_t, NUM_ITERATIONS>(mulKernel<uint32_t, NUM_ITERATIONS>, a_uint32_mul, b_uint32_mul);
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Testing 64-bit integer multiplications..." << std::endl;
    uint64_t a_uint64 = rand();
    uint64_t b_uint64 = rand();
    run_benchmark<uint64_t, NUM_ITERATIONS>(mulKernel<uint64_t, NUM_ITERATIONS>, a_uint64, b_uint64);
    std::cout << "----------------------------------------" << std::endl;
    
    std::cout << "Testing BB31 additions..." << std::endl;
    bb31_t a_bb31_add = bb31_t((int)rand());
    bb31_t b_bb31_add = bb31_t((int)rand());
    run_benchmark<bb31_t, NUM_ITERATIONS>(addKernel<bb31_t, NUM_ITERATIONS>, a_bb31_add, b_bb31_add);
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "Testing BB31 multiplications..." << std::endl;
    bb31_t a_bb31_mul = bb31_t((int)rand());
    bb31_t b_bb31_mul = bb31_t((int)rand());
    run_benchmark<bb31_t, NUM_ITERATIONS>(mulKernel<bb31_t, NUM_ITERATIONS>, a_bb31_mul, b_bb31_mul);
    std::cout << "----------------------------------------" << std::endl;

    return 0;
}
