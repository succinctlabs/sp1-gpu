#include <iostream>
#include <cuda_runtime.h>

// This kernel performs a specified number of uint32_t multiplications per thread.
// Each iteration does: x = a * x, which can be considered 1 operation.
__global__ void mulKernel(uint32_t *out, uint32_t a, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x = 1;

    // Unroll loops or use pragma unroll for more ops per iteration
    for (int i = 0; i < iterations; i++) {
        x = a * x; // 1 operation per iteration (1 mul)
    }

    // Write out the final value to ensure the compiler doesn't optimize away the computation
    out[idx] = x;
}

int main() {
    // GPU parameters
    int threadsPerBlock = 256;
    int numBlocks = 8192;  // Adjust to fully load your GPU
    int totalThreads = threadsPerBlock * numBlocks;

    // Number of iterations per thread
    // Increase this number to get a longer runtime and more stable measurements
    int iterations = 1000000;

    // Host and device pointers
    uint32_t *d_out;
    cudaMalloc((void**)&d_out, totalThreads * sizeof(uint32_t));
    cudaMemset(d_out, 0, totalThreads * sizeof(uint32_t));

    uint32_t a = 2;

    // Use CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up launch (optional) to remove first-time overheads
    mulKernel<<<numBlocks, threadsPerBlock>>>(d_out, a, iterations);
    cudaDeviceSynchronize();

    // Start timing
    cudaEventRecord(start);

    // Actual benchmark kernel launch
    mulKernel<<<numBlocks, threadsPerBlock>>>(d_out, a, iterations);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Compute operations
    double opsPerThread = (double)iterations; // 1 operation per iteration
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

    return 0;
}
