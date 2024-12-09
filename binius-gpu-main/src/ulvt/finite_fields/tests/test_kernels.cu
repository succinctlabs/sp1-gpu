#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <random>

#include "ulvt/finite_fields/baby_bear.cuh"
#include "ulvt/finite_fields/binary_tower.cuh"
#include "ulvt/finite_fields/binary_tower_simd.cuh"

__global__ void mul_binary_tower_32b_simd_ker(int size, uint32_t *a) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < size; i += stride) {
#pragma unroll
		for (int j = 0; j < 16; j++) {
			a[i] = mul_binary_tower_32b_simd<5>(a[i], a[i]);
		}
	}
}

__global__ void mul_binary_tower_16b_simd_ker(int size, uint32_t *a) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < size; i += stride) {
#pragma unroll
		for (int j = 0; j < 16; j++) {
			a[i] = mul_binary_tower_32b_simd<4>(a[i], a[i]);
		}
	}
}

__global__ void mul_binary_tower_8b_simd_ker(int size, uint32_t *a) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < size; i += stride) {
#pragma unroll
		for (int j = 0; j < 16; j++) {
			a[i] = mul_binary_tower_32b_simd<3>(a[i], a[i]);
		}
	}
}

__global__ void mul_baby_bear_ker(int size, BB31 *a) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < size; i += stride) {
		for (int j = 0; j < (1 << 10); j++) {
			a[i] *= a[i];
		}
	}
}

__global__ void inv_binary_tower_32b_ker(int size, uint32_t *a) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < size; i += stride) {
		for (int j = 0; j < 16; j++) {
			a[i] = FanPaarTowerField<5>::inverse(a[i]);
		}
	}
}

__global__ void inv_baby_bear_ker(int size, BB31 *a) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < size; i += stride) {
		for (int j = 0; j < 16; j++) {
			a[i] = BB31::inv(a[i]);
		}
	}
}

inline void print_errors() {
	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess) printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess) printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

constexpr unsigned int PARALLEL_OPS = 1 << 24;

TEST_CASE("mul_binary_tower_32b_simd_ker", "[mul_ker]") {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	std::mt19937 generator(seed);
	uint32_t *a = new uint32_t[PARALLEL_OPS];
	assert(a != nullptr);

	int blockSize;    // The launch configurator returned block size
	int minGridSize;  // The minimum grid size needed to achieve the
					  // maximum occupancy for a full device launch
	int gridSize;     // The actual grid size needed, based on input size
	cudaOccupancyMaxPotentialBlockSize<void(int, uint32_t *)>(
		&minGridSize, &blockSize, mul_binary_tower_32b_simd_ker, 0, 0
	);
	gridSize = (PARALLEL_OPS + blockSize - 1) / blockSize;

	printf("Launch configuration: GRIDS: %d, BLOCKS: %d\n", gridSize, blockSize);

	for (int i = 0; i < PARALLEL_OPS; i++) {
		a[i] = generator();
	}

	uint32_t *gpu_a;
	cudaMalloc(&gpu_a, PARALLEL_OPS * sizeof(uint32_t));
	cudaMemcpy(gpu_a, a, sizeof(uint32_t) * PARALLEL_OPS, cudaMemcpyHostToDevice);

	print_errors();

	mul_binary_tower_32b_simd_ker<<<gridSize, blockSize>>>(PARALLEL_OPS, gpu_a);

	print_errors();

	delete[] a;
	cudaFree(gpu_a);
}

TEST_CASE("mul_binary_tower_16b_simd_ker", "[mul_ker]") {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	std::mt19937 generator(seed);
	uint32_t *a = new uint32_t[PARALLEL_OPS];
	assert(a != nullptr);

	int blockSize;    // The launch configurator returned block size
	int minGridSize;  // The minimum grid size needed to achieve the
					  // maximum occupancy for a full device launch
	int gridSize;     // The actual grid size needed, based on input size
	cudaOccupancyMaxPotentialBlockSize<void(int, uint32_t *)>(
		&minGridSize, &blockSize, mul_binary_tower_32b_simd_ker, 0, 0
	);
	gridSize = (PARALLEL_OPS + blockSize - 1) / blockSize;

	printf("Launch configuration: GRIDS: %d, BLOCKS: %d\n", gridSize, blockSize);

	for (int i = 0; i < PARALLEL_OPS; i++) {
		a[i] = generator();
	}

	uint32_t *gpu_a;
	cudaMalloc(&gpu_a, PARALLEL_OPS * sizeof(uint32_t));
	cudaMemcpy(gpu_a, a, sizeof(uint32_t) * PARALLEL_OPS, cudaMemcpyHostToDevice);

	print_errors();

	mul_binary_tower_16b_simd_ker<<<gridSize, blockSize>>>(PARALLEL_OPS, gpu_a);

	print_errors();

	delete[] a;
	cudaFree(gpu_a);
}

TEST_CASE("mul_binary_tower_8b_simd_ker", "[mul_ker]") {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	std::mt19937 generator(seed);
	uint32_t *a = new uint32_t[PARALLEL_OPS];
	assert(a != nullptr);

	int blockSize;    // The launch configurator returned block size
	int minGridSize;  // The minimum grid size needed to achieve the
					  // maximum occupancy for a full device launch
	int gridSize;     // The actual grid size needed, based on input size
	cudaOccupancyMaxPotentialBlockSize<void(int, uint32_t *)>(
		&minGridSize, &blockSize, mul_binary_tower_32b_simd_ker, 0, 0
	);
	gridSize = (PARALLEL_OPS + blockSize - 1) / blockSize;

	printf("Launch configuration: GRIDS: %d, BLOCKS: %d\n", gridSize, blockSize);

	for (int i = 0; i < PARALLEL_OPS; i++) {
		a[i] = generator();
	}

	uint32_t *gpu_a;
	cudaMalloc(&gpu_a, PARALLEL_OPS * sizeof(uint32_t));
	cudaMemcpy(gpu_a, a, sizeof(uint32_t) * PARALLEL_OPS, cudaMemcpyHostToDevice);

	print_errors();

	mul_binary_tower_8b_simd_ker<<<gridSize, blockSize>>>(PARALLEL_OPS, gpu_a);

	print_errors();

	delete[] a;
	cudaFree(gpu_a);
}

TEST_CASE("mul_baby_bear_ker", "[mul_ker]") {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	std::mt19937 generator(seed);
	BB31 *a = new BB31[PARALLEL_OPS];
	assert(a != nullptr);

	int blockSize;    // The launch configurator returned block size
	int minGridSize;  // The minimum grid size needed to achieve the
					  // maximum occupancy for a full device launch
	int gridSize;     // The actual grid size needed, based on input size
	cudaOccupancyMaxPotentialBlockSize<void(int, BB31 *)>(&minGridSize, &blockSize, mul_baby_bear_ker, 0, 0);
	gridSize = (PARALLEL_OPS + blockSize - 1) / blockSize;

	printf("Launch configuration: GRIDS: %d, BLOCKS: %d\n", gridSize, blockSize);

	for (int i = 0; i < PARALLEL_OPS; i++) {
		a[i] = BB31(generator());
	}

	BB31 *gpu_a = nullptr;
	cudaMalloc(&gpu_a, PARALLEL_OPS * sizeof(BB31));
	cudaMemcpy(gpu_a, a, sizeof(BB31) * PARALLEL_OPS, cudaMemcpyHostToDevice);

	print_errors();

	mul_baby_bear_ker<<<gridSize, blockSize>>>(PARALLEL_OPS, gpu_a);

	print_errors();

	delete[] a;
	cudaFree(gpu_a);
}

TEST_CASE("inv_binary_tower_32b_ker", "[inv_ker]") {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	std::mt19937 generator(seed);
	uint32_t *a = new uint32_t[PARALLEL_OPS];
	assert(a != nullptr);

	int blockSize;    // The launch configurator returned block size
	int minGridSize;  // The minimum grid size needed to achieve the
					  // maximum occupancy for a full device launch
	int gridSize;     // The actual grid size needed, based on input size
	cudaOccupancyMaxPotentialBlockSize<void(int, uint32_t *)>(&minGridSize, &blockSize, inv_binary_tower_32b_ker, 0, 0);
	gridSize = (PARALLEL_OPS + blockSize - 1) / blockSize;

	printf("Launch configuration: GRIDS: %d, BLOCKS: %d\n", gridSize, blockSize);

	for (int i = 0; i < PARALLEL_OPS; i++) {
		a[i] = generator();
	}

	uint32_t *gpu_a;
	cudaMalloc(&gpu_a, PARALLEL_OPS * sizeof(uint32_t));
	cudaMemcpy(gpu_a, a, sizeof(uint32_t) * PARALLEL_OPS, cudaMemcpyHostToDevice);

	print_errors();

	inv_binary_tower_32b_ker<<<gridSize, blockSize>>>(PARALLEL_OPS, gpu_a);

	print_errors();

	delete[] a;
	cudaFree(gpu_a);
}

TEST_CASE("inv_baby_bear_ker", "[inv_ker]") {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	std::mt19937 generator(seed);
	BB31 *a = new BB31[PARALLEL_OPS];

	assert(a != nullptr);

	int blockSize;    // The launch configurator returned block size
	int minGridSize;  // The minimum grid size needed to achieve the
					  // maximum occupancy for a full device launch
	int gridSize;     // The actual grid size needed, based on input size
	cudaOccupancyMaxPotentialBlockSize<void(int, BB31 *)>(&minGridSize, &blockSize, inv_baby_bear_ker, 0, 0);
	gridSize = (PARALLEL_OPS + blockSize - 1) / blockSize;

	printf("Launch configuration: GRIDS: %d, BLOCKS: %d\n", gridSize, blockSize);

	for (int i = 0; i < PARALLEL_OPS; i++) {
		a[i] = BB31(generator());
	}

	BB31 *gpu_a = nullptr;
	cudaMalloc(&gpu_a, PARALLEL_OPS * sizeof(BB31));
	cudaMemcpy(gpu_a, a, sizeof(BB31) * PARALLEL_OPS, cudaMemcpyHostToDevice);

	print_errors();

	inv_baby_bear_ker<<<gridSize, blockSize>>>(PARALLEL_OPS, gpu_a);

	print_errors();

	delete[] a;
	cudaFree(gpu_a);
}
