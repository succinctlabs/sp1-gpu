#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include "ulvt/utils/common.cuh"

enum class DataOrder : char { INVALID = -1, IN_ORDER, BIT_REVERSED };

template <typename E>
class NTTData {
public:
	DataOrder order;
	std::unique_ptr<E[]> data;
	size_t size;

	NTTData(DataOrder order, size_t size) : order(order), size(size), data(std::make_unique<E[]>(size)) {}
	NTTData(size_t size) : order(DataOrder::INVALID), size(size), data(std::make_unique<E[]>(size)) {}
	inline size_t byte_len() const { return sizeof(E) * size; }
};

extern dim3 KERNEL_LAUNCH_CONFIGS[31][3][2];

template <typename E>
class NTTConfRad2 {
public:
	E generator;
	uint32_t log_group_order;
	int log_inp_size;

	NTTConfRad2(E gen, uint32_t log_grp_order, int inp_log_size)
		: generator(gen), log_group_order(log_grp_order), log_inp_size(inp_log_size) {
		ASSERT(inp_log_size >= 1);
		// NOTE: Even though you can do 27 log inputs the kernel cannot support than 2^16-1 in y, z dimensions of the
		// grid
		ASSERT(inp_log_size <= 27);
		ASSERT(log_grp_order >= inp_log_size);
	}

	// Returns a pair with the data and the number of kernels to launch
	// dim3[kern][1] contains the gridDim and dim3[kern][0] contains the blockDim
	std::pair<dim3 (*)[2], int> get_kernel_launch_confs() {
		int num_kerns = 1 + (log_inp_size - 1) / MAX_STAGES_PER_KERNEL;
		return std::make_pair(KERNEL_LAUNCH_CONFIGS[log_inp_size], num_kerns);
	}
};

template <typename T, typename P>
class AdditiveNTTConf {
public:
	int log_h;
	int log_rate;

	AdditiveNTTConf(int log_h, int log_rate) : log_h(log_h), log_rate(log_rate) {
		ASSERT(log_h >= 1);
		ASSERT(log_h + log_rate <= P::N_BITS());

		ASSERT(log_rate >= 0 && log_rate <= 4);
	}

	// Returns a pair with the data and the number of kernels to launch, since we are using dit ntt, you have to index
	// the kernels from num_kerns - 1, ... 0
	// dim3[kern][1] contains the gridDim and dim3[kern][0] contains the blockDim
	std::pair<dim3 (*)[2], int> get_kernel_launch_confs() {
		const int max_stages_per_kernel = MAX_STAGES_PER_KERNEL - log_rate;
		int num_kerns = 1 + (log_h - 1) / max_stages_per_kernel;
		// For now we only support num_kerns of atmost 3
		ASSERT(num_kerns <= 3);
		if (log_rate == 0) {
			return std::make_pair(KERNEL_LAUNCH_CONFIGS[log_h], num_kerns);
		}

		if (num_kerns >= 3) {
			auto dim_blocks_z = log_rate;
			auto dim_blocks_x = std::min(10 - log_rate, log_h - 1 - max_stages_per_kernel * 2);
			auto dim_blocks_y = std::max(0, 10 - log_rate - dim_blocks_x);

			auto dim_grids_z = std::min(15, std::max(0, log_h - 1 - dim_blocks_x - dim_blocks_y));
			auto dim_grids_y = std::max(0, log_h - 1 - dim_blocks_x - dim_blocks_y - dim_grids_z);

			dim3 dim_blocks(1 << dim_blocks_x, 1 << dim_blocks_y, 1 << dim_blocks_z);
			dim3 dim_grids(1, 1 << dim_grids_y, 1 << dim_grids_z);

			kernel_launch_conf[2][0] = dim3(1 << dim_blocks_x, 1 << dim_blocks_y, 1 << dim_blocks_z);
			kernel_launch_conf[2][1] = dim3(1, 1 << dim_grids_y, 1 << dim_grids_z);
		}
		if (num_kerns >= 2) {
			auto dim_blocks_z = log_rate;
			auto dim_blocks_x = std::min(10 - log_rate, log_h - 1 - max_stages_per_kernel);
			auto dim_blocks_y = std::max(0, 10 - log_rate - dim_blocks_x);

			auto dim_grids_y = std::min(max_stages_per_kernel, std::max(0, log_h - 1 - dim_blocks_x - dim_blocks_y));
			auto dim_grids_z = std::max(0, log_h - 1 - dim_blocks_x - dim_blocks_y - dim_grids_y);

			kernel_launch_conf[1][0] = dim3(1 << dim_blocks_x, 1 << dim_blocks_y, 1 << dim_blocks_z);
			kernel_launch_conf[1][1] = dim3(1, 1 << dim_grids_y, 1 << dim_grids_z);
		}
		{
			auto dim_blocks_z = log_rate;
			auto dim_blocks_x = std::min(10 - log_rate, log_h - 1);

			auto dim_grids_x = std::max(0, log_h - dim_blocks_x - 1);

			kernel_launch_conf[0][0] = dim3(1 << dim_blocks_x, 1, 1 << dim_blocks_z);
			kernel_launch_conf[0][1] = dim3(1 << dim_grids_x);
		}

		return std::make_pair(kernel_launch_conf, num_kerns);
	}

private:
	dim3 kernel_launch_conf[3][2];
};
