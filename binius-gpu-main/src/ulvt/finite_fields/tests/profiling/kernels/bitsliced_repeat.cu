#include <cstdint>

#include "../../../circuit_generator/unrolled/binary_tower_unrolled.cuh"
#include "bitsliced_repeat.cuh"

__global__ void bitsliced_repeat(uint32_t* x, uint32_t* y, uint32_t* dst) {
	const int TOWER_HEIGHT = 5;
	const int BITS_WIDTH = 1 << TOWER_HEIGHT;

	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	uint32_t x_local[BITS_WIDTH];
	uint32_t y_local[BITS_WIDTH];
	uint32_t dst_local[BITS_WIDTH];

	for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
		// Every thread multiplies 32 items of size BITS_WIDTH in parallel
		x_local[i] = x[i + BITS_WIDTH * tid];
		y_local[i] = y[i + BITS_WIDTH * tid];
	}

	for (uint32_t i = 0; i < 50000; ++i) {
		multiply_unrolled<TOWER_HEIGHT>(x_local, y_local, dst_local);

		for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
			x_local[i] = dst_local[i];
		}
	}

	for (uint32_t i = 0; i < BITS_WIDTH; ++i) {
		dst[i + BITS_WIDTH * tid] = dst_local[i];
	}
}