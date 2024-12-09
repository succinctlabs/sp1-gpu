#include <cstdint>
#include <iostream>
#include <nvbench/nvbench.cuh>

#include "./kernels/babybear_repeat.cuh"
#include "./kernels/bitsliced_repeat.cuh"

void bitsliced(nvbench::state& state) {
	uint32_t threads_per_block = 256;
	uint32_t blocks = 256;

	uint32_t threads = threads_per_block * blocks;

	uint32_t *a, *b, *dst;

	cudaMalloc(&a, 32 * threads * sizeof(uint32_t));
	cudaMalloc(&b, 32 * threads * sizeof(uint32_t));
	cudaMalloc(&dst, 32 * threads * sizeof(uint32_t));

	state.exec([a, b, dst, threads_per_block, blocks](nvbench::launch& launch) {
		bitsliced_repeat<<<blocks, threads_per_block>>>(a, b, dst);
	});

	cudaDeviceSynchronize();

	cudaFree(a);
	cudaFree(b);
	cudaFree(dst);
}

void babybear(nvbench::state& state) {
	uint32_t threads_per_block = 256;
	uint32_t blocks = 256;

	uint32_t threads = threads_per_block * blocks;

	BB31 *operands, *results;

	cudaMalloc(&operands, threads * sizeof(BB31));
	cudaMalloc(&results, threads * sizeof(BB31));

	state.exec([operands, results, threads_per_block, blocks](nvbench::launch& launch) {
		babybear_repeat<<<blocks, threads_per_block>>>(operands, results);
	});

	cudaDeviceSynchronize();

	cudaFree(operands);
	cudaFree(results);
}

NVBENCH_BENCH(bitsliced);

NVBENCH_BENCH(babybear);

int main(int argc, char** argv) {
	NVBENCH_MAIN_BODY(argc, argv);
	return 0;
}
