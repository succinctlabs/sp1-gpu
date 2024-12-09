#include "../../../baby_bear.cuh"

__global__ void babybear_repeat(BB31* operands, BB31* results) {
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	BB31 a = operands[tid];
	BB31 b = a;

	for (int i = 0; i < 32 * 50000; ++i) {
		a = b * a;
	}

	results[tid] = a;
}