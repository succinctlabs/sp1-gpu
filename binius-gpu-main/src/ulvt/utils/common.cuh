#pragma once

#ifdef STRIP_ASSERTIONS
#define ASSERT(x) (x)
#else
#include <cstdlib>
#define ASSERT(x)                                                                                                      \
	{                                                                                                                  \
		if (!(x)) {                                                                                                    \
			std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__;                                        \
			std::cerr << " inside " << __FUNCTION__ << std::endl;                                                      \
			std::cerr << "Condition: " << #x << std::endl;                                                             \
			abort();                                                                                                   \
		}                                                                                                              \
	}
#endif

#ifdef STRIP_CUDA_CHECK
#define CUDA_CHECK(x) (x)
#else
#define CUDA_CHECK(x)                                                                                                  \
	{                                                                                                                  \
		if ((x) != cudaSuccess) {                                                                                      \
			std::cerr << "Cuda error at " << __FILE__ << ":" << __LINE__;                                              \
			std::cerr << " inside " << __FUNCTION__ << std::endl;                                                      \
			std::cerr << "Error: " << cudaGetErrorString(x) << std::endl;                                              \
		}                                                                                                              \
	}
#endif

#define MAX_SHARED_MEM (1 << 15)
#define MAX_STAGES_PER_KERNEL 11

bool check_gpu_capabilities();
