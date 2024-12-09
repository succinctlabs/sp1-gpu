#pragma once

#include <cstdio>
#include <vector>

#include "nttconf.cuh"
#include "ulvt/utils/common.cuh"

// inplace additive ntt butterfly
template <typename T, typename P>
static constexpr __device__ void antt_butterfly(T& u, T& v, T w) {
	u = P::add(u, P::multiply(w, v));
	v = P::add(u, v);
}

template <typename T, typename P>
static constexpr __device__ __host__ T subspace_map(const T element, const T constant) {
	return P::add(P::square(element), P::multiply(constant, element));
}

static constexpr __device__ int get_v_offset(const int uidx, const int stage) { return uidx | (1 << stage); }

static constexpr __device__ int get_u_offset(const int stage, const int butterfly_block, const int butterfly_idx) {
	return butterfly_block << (stage + 1) | butterfly_idx;
}

// the j variable from the model
static constexpr __device__ int get_butterfly_block(const int thread_id, const int stage) {
	return thread_id / (1 << stage);
}

// the k variable from the model
static constexpr __device__ int get_butterfly(const int thread_id, const int stage) { return thread_id % (1 << stage); }

template <typename T>
static constexpr __device__ __host__ T* _flat_array_2d(
	T* _2d_data, const size_t width_bytes, const int row, const int col
) {
	return ((T*)((char*)_2d_data + row * width_bytes) + col);
}

// For convenience this is operating on sizeof(T) widths as opposed to cuda's byte widths
template <typename T>
static constexpr __device__ __host__ T& flat_array_3d(
	T* _3d_data, const int width, const int height, const int x, const int y, const int z
) {
	return _3d_data[x + width * (y + z * height)];
}

template <typename T>
static constexpr __device__ __host__ T& flat_array_2d(
	T* _2d_data, const size_t width_bytes, const int row, const int col
) {
	return *_flat_array_2d(_2d_data, width_bytes, row, col);
}

static constexpr __device__ bool is_bit_set(const int x, const int i) { return (x >> i) & 1; }

template <typename T, typename P>
static constexpr __device__ T calculate_twiddle(
	const T* constants,
	const size_t c_pitch,
	const int log_h,
	const int log_rate,
	const int coset,
	const int stage,
	const int butterfly_block
) {
	T sum = P::ZERO();
	for (int k = 0; k < log_h + log_rate - 1 - stage; k++) {
		int indicator = coset << (log_h - 1 - stage) | butterfly_block;
		if (is_bit_set(indicator, k)) {
			sum = P::add(sum, flat_array_2d(constants, c_pitch, stage, k));
		}
	}
	return sum;
}

template <typename T>
struct AdditiveNTTKernelParams {
	T* data_io;
	size_t data_pitch;
	T* constants;
	size_t constants_pitch;
	int log_h;
	int log_rate;
	int start_stage;
	int end_stage;
};

template <typename T, typename P>
static __global__ void additive_ntt_kernel(AdditiveNTTKernelParams<T> kernel_params) {
	__shared__ char shared_mem[MAX_SHARED_MEM];
	T* data_io = kernel_params.data_io;
	size_t d_pitch = kernel_params.data_pitch;
	const T* pre_computed = kernel_params.constants;
	size_t c_pitch = kernel_params.constants_pitch;
	int log_h = kernel_params.log_h;
	int log_rate = kernel_params.log_rate;
	int start_stage = kernel_params.start_stage;
	int end_stage = kernel_params.end_stage;

	T* uv_mem = (T*)shared_mem;

	const int local_id = threadIdx.x;
	const int coset = threadIdx.z;
	const int max_stages_per_kernel = MAX_STAGES_PER_KERNEL - log_rate;

	int unit_vec[3] = {0};
	unit_vec[start_stage / max_stages_per_kernel] = 1;

	const int exec_id_1 = local_id + blockDim.x * blockIdx.x;

	const int exec_id_2 = threadIdx.x * gridDim.y * blockDim.y + blockIdx.y +
						  gridDim.y * blockDim.y * blockDim.x * blockIdx.z + gridDim.y * threadIdx.y;

	const int exec_id_3 = threadIdx.x * gridDim.z * gridDim.y * blockDim.y + blockIdx.z + gridDim.z * blockIdx.y +
						  gridDim.z * gridDim.y * threadIdx.y;

	const int exec_id = unit_vec[0] * exec_id_1 + unit_vec[1] * exec_id_2 + unit_vec[2] * exec_id_3;

	const int local_off = blockDim.x;

	const int uv_width = blockDim.x * 2;
	const int uv_height = blockDim.y;

	const int butterfly_block = get_butterfly_block(exec_id, end_stage - 1);
	const int butterfly_idx = get_butterfly(exec_id, end_stage - 1);
	const int uoff = get_u_offset(end_stage - 1, butterfly_block, butterfly_idx);
	const int voff = get_v_offset(uoff, end_stage - 1);

	// copy from global memory into shared memory here
	flat_array_3d<T>(uv_mem, uv_width, uv_height, threadIdx.x, threadIdx.y, coset) =
		flat_array_2d<T>(data_io, d_pitch, coset, uoff);
	flat_array_3d<T>(uv_mem, uv_width, uv_height, threadIdx.x + local_off, threadIdx.y, coset) =
		flat_array_2d<T>(data_io, d_pitch, coset, voff);

	for (int stage = end_stage - 1; stage >= start_stage; stage--) {
		// calculate twiddle, this stage has to be from the global context
		int butterfly_block_global = get_butterfly_block(exec_id, stage);
		T twiddle =
			calculate_twiddle<T, P>(pre_computed, c_pitch, log_h, log_rate, coset, stage, butterfly_block_global);

		// These stages have to be from the local context
		int butterfly_block = get_butterfly_block(local_id, stage - start_stage);
		int butterfly_idx = get_butterfly(local_id, stage - start_stage);
		int uoff_local = get_u_offset(stage - start_stage, butterfly_block, butterfly_idx);
		int voff_local = get_v_offset(uoff_local, stage - start_stage);
		T& u = flat_array_3d<T>(uv_mem, uv_width, uv_height, uoff_local, threadIdx.y, coset);
		T& v = flat_array_3d<T>(uv_mem, uv_width, uv_height, voff_local, threadIdx.y, coset);
		antt_butterfly<T, P>(u, v, twiddle);

		__syncthreads();
	}

	flat_array_2d<T>(data_io, d_pitch, coset, uoff) =
		flat_array_3d<T>(uv_mem, uv_width, uv_height, threadIdx.x, threadIdx.y, coset);
	flat_array_2d<T>(data_io, d_pitch, coset, voff) =
		flat_array_3d<T>(uv_mem, uv_width, uv_height, threadIdx.x + local_off, threadIdx.y, coset);
}

static constexpr void print_kern_launch(dim3 dim_grids, dim3 dim_blocks, int kern) {
	printf(
		"Kernel %d launch configuration blocks: (%d, %d, %d) grids: (%d, %d, %d)\n",
		kern,
		dim_blocks.x,
		dim_blocks.y,
		dim_blocks.z,
		dim_grids.x,
		dim_grids.y,
		dim_grids.z
	);
}

template <typename T, typename P>
class AdditiveNTT {
public:
	AdditiveNTT(const AdditiveNTTConf<T, P>& nttconf) : ntt_conf(nttconf) {
		const int input_size = 1 << ntt_conf.log_h;
		const int output_size = 1 << (ntt_conf.log_h + ntt_conf.log_rate);

		auto s_evals = precompute_subspace_evals();
		auto largest_width = ntt_conf.log_h + ntt_conf.log_rate - 1;
		CUDA_CHECK(cudaMallocPitch(&pre_computed, &constants_pitch, sizeof(T) * largest_width, ntt_conf.log_h));

		CUDA_CHECK(cudaMemcpy2D(
			pre_computed,
			constants_pitch,
			s_evals,
			largest_width * sizeof(T),
			largest_width * sizeof(T),
			ntt_conf.log_h,
			cudaMemcpyHostToDevice
		));

		delete[] s_evals;

		CUDA_CHECK(cudaMallocPitch(&data_in_out, &out_pitch, sizeof(T) * input_size, 1 << ntt_conf.log_rate));
	}

	bool apply(const NTTData<T>& input, NTTData<T>& output) {
		auto log_h = ntt_conf.log_h;
		auto log_rate = ntt_conf.log_rate;
		size_t input_size = 1 << log_h;
		size_t output_size = 1 << (log_h + log_rate);
		if (input.size != input_size || input.order != DataOrder::IN_ORDER) {
			return false;
		}

		char* data_io = (char*)data_in_out;
		// copy data into output buffer first, which will operated on by the kernel
		// This copies the address of input 2^log_rate times into our temporary buffer
		for (size_t i = 0; i < (1 << log_rate); i++) {
			CUDA_CHECK(cudaMemcpy(&data_io[i * out_pitch], input.data.get(), input.byte_len(), cudaMemcpyHostToDevice));
		}

		// One key difference between the bb31 and this implementation is that it has a fixed maximum stage of 11 per
		// kernel This however has a maximum stage limit of 11-log_rate per kernel

		// Another thing to note is that the kernel launches has to be reversed.
		// For data of size 2^14, the second kernel does 13-9 and first kernel does 8-0
		const int max_stages_per_kernel = MAX_STAGES_PER_KERNEL - log_rate;
		int stages = log_h;
		AdditiveNTTKernelParams<T> kernel_params;
		kernel_params.data_io = data_in_out;
		kernel_params.data_pitch = out_pitch;
		kernel_params.constants = pre_computed;
		kernel_params.constants_pitch = constants_pitch;
		kernel_params.log_h = log_h;
		kernel_params.log_rate = log_rate;

		auto [kernel_launch_conf, num_kerns] = ntt_conf.get_kernel_launch_confs();
		for (int kern = num_kerns - 1; kern >= 0; kern--) {
			// int
			kernel_params.start_stage = kern * max_stages_per_kernel;
			kernel_params.end_stage = std::min(stages, max_stages_per_kernel * (kern + 1));
#ifndef NDEBUG
			print_kern_launch(kernel_launch_conf[kern][1], kernel_launch_conf[kern][0], kern);
			printf(
				"Kernel %d (..., start_stage = %d, end_stage = %d)\n",
				kern,
				kernel_params.start_stage,
				kernel_params.end_stage
			);
#endif  // DEBUG
			additive_ntt_kernel<T, P><<<kernel_launch_conf[kern][1], kernel_launch_conf[kern][0]>>>(kernel_params);
		}

		// At the end copy back the data from the output into a single contiguous memory
		output.order = DataOrder::IN_ORDER;
		CUDA_CHECK(cudaMemcpy2D(
			output.data.get(),
			input.byte_len(),
			data_in_out,
			out_pitch,
			input.byte_len(),
			1 << log_rate,
			cudaMemcpyDeviceToHost
		));

		// Possibly Remove this
		CUDA_CHECK(cudaDeviceSynchronize());

		return true;
	}

	~AdditiveNTT() {
		CUDA_CHECK(cudaFree(pre_computed));
		CUDA_CHECK(cudaFree(data_in_out));
	}

private:
	inline T* precompute_subspace_evals() const {
		auto largest_width = ntt_conf.log_h + ntt_conf.log_rate - 1;
		auto pitch = largest_width * sizeof(T);
		T* constants = new T[ntt_conf.log_h * largest_width];

		std::vector<T> norm_consts;
		norm_consts.reserve(ntt_conf.log_h);

		for (int i = 1; i < ntt_conf.log_rate + ntt_conf.log_h; i++) {
			flat_array_2d(constants, pitch, 0, i - 1) = T(1 << i);
		}
		norm_consts.push_back(P::ONE());

		for (int i = 1; i < ntt_conf.log_h; i++) {
			T norm_prev = norm_consts.back();
			T* s_evals_prev = _flat_array_2d(constants, pitch, i - 1, 0);

			T norm_const_i = subspace_map<T, P>(s_evals_prev[0], norm_prev);

			for (size_t j = 1; j < ntt_conf.log_h + ntt_conf.log_rate - i; j++) {
				T sij_prev = s_evals_prev[j];
				flat_array_2d(constants, pitch, i, j - 1) = subspace_map<T, P>(sij_prev, norm_prev);
			}

			norm_consts.push_back(norm_const_i);
		}

		for (size_t i = 0; i < ntt_conf.log_h; i++) {
			T inv_norm_const = P::inverse(norm_consts[i]);
			T* si_evals = _flat_array_2d(constants, pitch, i, 0);
			for (size_t j = 0; j < ntt_conf.log_h + ntt_conf.log_rate - i - 1; j++) {
				si_evals[j] = P::multiply(inv_norm_const, si_evals[j]);
			}
		}

		return constants;
	}

	size_t constants_pitch;
	size_t out_pitch;
	AdditiveNTTConf<T, P> ntt_conf;
	// host memory pointers

	// gpu memory pointers
	T* pre_computed;
	T* data_in_out;
};
