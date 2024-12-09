#include <iostream>

#include "common.cuh"

// TODO: In the future we need to move this into enum to return specific errors
bool check_gpu_capabilities() {
	int nDevices;

	CUDA_CHECK(cudaGetDeviceCount(&nDevices));

	if (nDevices < 1) {
		std::cerr << "Capabilities Error: There are no cuda capable devices found "
					 "on this machine"
				  << std::endl;
		return false;
	}

	// Assuming the first device is our device
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

	if (prop.maxThreadsPerBlock < 1024) {
		std::cerr << "Capabilities Error: less than 1024 threads available per block" << std::endl;
		return false;
	}

	if (prop.sharedMemPerBlock <= (1 << 15)) {
		std::cerr << "Capabilities Error: less than 32kb of shared memory available" << std::endl;
		return false;
	}

	if (prop.maxThreadsDim[0] < 1024 || prop.maxThreadsDim[1] < 1024) {
		std::cerr << "Capabilities Error: less than 1024 threads for x,y dimensions" << std::endl;
		return false;
	}

	if (prop.maxGridSize[0] < (1 << 20) || prop.maxGridSize[1] < (1 << 12) || prop.maxGridSize[2] < (1 << 15)) {
		std::cerr << "Capabilities Error: x,y,z grid dimensions too low" << std::endl;
		return false;
	}

	return true;
}
