#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

#include "nttconf.cuh"
#include "ulvt/utils/common.cuh"

static inline __host__ __device__ int reverse_bits(uint32_t n, const int size) {
	int ans = 0;
	for (int i = (size - 1); i >= 0; i--) {
		ans |= (n & 1) << i;
		n >>= 1;
	}
	return ans;
}

// Given an in_out array of size `n_size`, this kernel would do an inplace reversal assuming bit length is idx_size
template <typename T>
static __global__ void bit_reverse_in_place_ker(T* in_out, const uint32_t in_size, const uint32_t idx_size) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	for (int i = index; i < in_size; i += stride) {
		int r_idx = reverse_bits(i, idx_size);
		if (r_idx < i) {
			T tmp = in_out[i];
			in_out[i] = in_out[r_idx];
			in_out[r_idx] = tmp;
		}
	}
}

// Computes the butterfly in place
// U = u + v
// V = (u - v) * w
template <typename T>
static constexpr __device__ void dif_butterfly(T& u, T& v, const T w) {
	T v_tmp = u - v;
	u = u + v;
	v = v_tmp * w;
}

static constexpr __device__ int get_v_offset(const int stage) { return (1 << stage); }

template <typename T>
static constexpr __device__ T& get_v_from_u_stage(T* u, const int stage) {
	return u[get_v_offset(stage)];
}

template <typename T>
static constexpr __device__ T& get_u_for_stage(T* uv_mem, const int thread_uid, const int stage) {
	const int off = ((thread_uid % (1 << stage)) | (thread_uid >> stage) << (stage + 1));
	return uv_mem[off];
}

template <typename T>
static constexpr __device__ T get_tw_from_stage(T* tw_mem, const int exec_uid, const int stage, const int twiddle_size) {
	const int off = (exec_uid / (1 << stage)) % (twiddle_size / (1 << stage));
	return tw_mem[off];
}

template <typename T>
static __global__ void ntt_kernel(T* in_data, T* twiddles, const int log_inp_len, const int start_stage, const int end_stage) {
	// This is the maximum size of the shared memory, since we are launching 2^10 threads which is hardcoded
	// it'll never fill more than 2^11 * 4 = 2^13 bytes of data
	__shared__ char shared_mem[MAX_SHARED_MEM];

	T* uv_mem = (T*)(shared_mem);

	const int twiddle_size = 1 << (log_inp_len - 1);

	const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
	const int stride_3 = 1 << 22;
	const int exec_id_3 =
		threadIdx.x * stride_3 + blockIdx.z + gridDim.z * blockIdx.y + gridDim.z * gridDim.y * threadIdx.y;

	const int uoff_3 =
		(threadIdx.x * 2) * stride_3 + blockIdx.z + gridDim.z * blockIdx.y + gridDim.z * gridDim.y * threadIdx.y;
	const int voff_3 =
		(threadIdx.x * 2 + 1) * stride_3 + blockIdx.z + gridDim.z * blockIdx.y + gridDim.z * gridDim.y * threadIdx.y;

	// This is 2^11
	const int stride_2 = gridDim.y * blockDim.y;
	const int exec_id_2 = threadIdx.x * stride_2 + blockIdx.y + gridDim.y * blockDim.y * blockDim.x * blockIdx.z +
						  gridDim.y * threadIdx.y;

	const int uoff_2 = (threadIdx.x * 2) * stride_2 + blockIdx.y +
					   gridDim.y * blockDim.y * blockDim.x * 2 * blockIdx.z + gridDim.y * threadIdx.y;
	const int voff_2 = (threadIdx.x * 2 + 1) * stride_2 + blockIdx.y +
					   gridDim.y * blockDim.y * blockDim.x * 2 * blockIdx.z + gridDim.y * threadIdx.y;

	const int exec_id_1 = thread_id + blockIdx.x * blockDim.x;
	const int uoff_1 = exec_id_1 * 2;
	const int voff_1 = exec_id_1 * 2 + 1;

	int vec[3] = {0};
	vec[start_stage / MAX_STAGES_PER_KERNEL] = 1;

	const int uoff = vec[0] * uoff_1 + vec[1] * uoff_2 + vec[2] * uoff_3;
	const int voff = vec[0] * voff_1 + vec[1] * voff_2 + vec[2] * voff_3;
	const int exec_id = vec[0] * exec_id_1 + vec[1] * exec_id_2 + vec[2] * exec_id_3;

	T* u = &in_data[uoff];
	T* v = &in_data[voff];
	uv_mem[thread_id * 2] = *u;
	uv_mem[thread_id * 2 + 1] = *v;

	for (int stage = start_stage; stage < end_stage; stage++) {
		T& u = get_u_for_stage<T>(uv_mem, thread_id, stage - start_stage);
		T& v = get_v_from_u_stage<T>(&u, stage - start_stage);
		T w = get_tw_from_stage<T>(twiddles, exec_id, stage, twiddle_size);

		dif_butterfly<T>(u, v, w);
		__syncthreads();
	}

	u = &in_data[uoff];
	v = &in_data[voff];
	*u = uv_mem[thread_id * 2];
	*v = uv_mem[thread_id * 2 + 1];
}

template <typename E>
class NTT {
public:
	NTT(const NTTConfRad2<E>& nttconf) : ntt_conf(nttconf) {
		// Maximum supported size of our primitive for now.
		static_assert(sizeof(E) <= 4, "As of now the maximum size of your field can atmost be 4 bytes");
		// Initialize twiddles here.
		auto twiddles = pre_compute();

		CUDA_CHECK(cudaMalloc(&twiddles_gpu, sizeof(E) * twiddles.size()));
		CUDA_CHECK(cudaMemcpy(twiddles_gpu, twiddles.data(), sizeof(E) * twiddles.size(), cudaMemcpyHostToDevice));

		// Change twiddles to bit reversed order and store in global memory
		const int blockSize = 512;
		const int gridSize = (twiddles.size() + blockSize - 1) / blockSize;
		const int idx_size = ntt_conf.log_inp_size - 1;
		bit_reverse_in_place_ker<E><<<gridSize, blockSize>>>(twiddles_gpu, twiddles.size(), idx_size);

		CUDA_CHECK(cudaDeviceSynchronize());

		CUDA_CHECK(cudaMalloc(&input_global, sizeof(E) * (1 << ntt_conf.log_inp_size)));
	}

	~NTT() {
		// Unintialize twiddles and general cleanup

		CUDA_CHECK(cudaFree(twiddles_gpu));
		CUDA_CHECK(cudaFree(input_global));
	}

	void apply(const NTTData<E>& input, NTTData<E>& output) {
		// First step is to copy the input data to global memory
		size_t input_size = 1 << ntt_conf.log_inp_size;
		ASSERT(input_size == input.size);
		CUDA_CHECK(cudaMemcpy(input_global, input.data.get(), input.byte_len(), cudaMemcpyHostToDevice));

		// We expect our input data to be in bit reversed order
		if (input.order != DataOrder::BIT_REVERSED) {
			const int blockSize = 512;
			const int gridSize = (input_size + blockSize - 1) / blockSize;
			bit_reverse_in_place_ker<E><<<gridSize, blockSize>>>(input_global, input_size, ntt_conf.log_inp_size);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		const auto log_inp_len = ntt_conf.log_inp_size;
		auto [kernel_launch_conf, num_kerns] = ntt_conf.get_kernel_launch_confs();
		int start_stage = 0;
		for (int kern = 0; kern < num_kerns; kern++) {
			int end_stage = std::min(start_stage + MAX_STAGES_PER_KERNEL, log_inp_len);
			ntt_kernel<<<kernel_launch_conf[kern][1], kernel_launch_conf[kern][0]>>>(
				input_global, twiddles_gpu, ntt_conf.log_inp_size, start_stage, end_stage
			);
			start_stage += MAX_STAGES_PER_KERNEL;
		}

		output.order = DataOrder::IN_ORDER;
		ASSERT(output.size == input_size);
		CUDA_CHECK(cudaMemcpy(output.data.get(), input_global, output.byte_len(), cudaMemcpyDeviceToHost));
	}

private:
	inline std::vector<E> pre_compute() const {
		const size_t num_twiddles = 1 << (ntt_conf.log_inp_size - 1);
		std::vector<E> twiddles;
		twiddles.reserve(num_twiddles);

		const uint32_t omega_order = 1 << (ntt_conf.log_group_order - ntt_conf.log_inp_size);
		E w = E::pow(ntt_conf.generator, omega_order);
		E prev = E::one();

		twiddles.push_back(prev);

		for (size_t i = 1; i < num_twiddles; i++) {
			prev = prev * w;
			twiddles.push_back(prev);
		}
		ASSERT(E::pow(prev * w, 2) == E::one());

		return twiddles;
	}

	NTTConfRad2<E> ntt_conf;
	E* twiddles_gpu;
	E* input_global;
};
