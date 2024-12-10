#include <iostream>
#include <cuda_runtime.h>

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
void run_benchmark(void (*opKernel)(T *out, T a, T b), T a, T b) {
        // GPU parameters
    int threadsPerBlock = 256;
    int numBlocks = 8192;  // Adjust to fully load your GPU
    int totalThreads = threadsPerBlock * numBlocks;

    // Host and device pointers
    T *d_out;
    cudaMalloc((void**)&d_out, totalThreads * sizeof(T));
    cudaMemset(d_out, 0, totalThreads * sizeof(T));

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

    uint32_t a = 2;
    uint32_t b = 1;
    run_benchmark<uint32_t, 1000000>(mulKernel<uint32_t, 1000000>, a, b);
    return 0;
}
