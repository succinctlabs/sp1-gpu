#pragma once

#include <array>
#include <cstdint>
#include <cstring>

template <size_t BITSLICING_BITS_WIDTH>
class BitsliceUtils {
private:
	std::array<uint32_t, BITSLICING_BITS_WIDTH>& buffer;

	static constexpr size_t INTS_PER_UNBITSLICED_VALUE = BITSLICING_BITS_WIDTH/32;

	__host__ __device__ static void transpose32(uint32_t A[32]) {
		int j, k;
		uint32_t m, t;

		m = 0x0000FFFF;
		for (j = 16; j != 0; j = j >> 1, m = m ^ (m << j)) {
			for (k = 0; k < 32; k = (k + j + 1) & ~j) {
				t = ((A[k] >> j) ^ (A[k + j])) & m;
				A[k] = A[k] ^ (t << j);
				A[k + j] = A[k + j] ^ (t);
			}
		}
	}

public:
	BitsliceUtils(std::array<uint32_t, BITSLICING_BITS_WIDTH>& data) : buffer(data) {}
	~BitsliceUtils() {}

	__host__ __device__ static void bitslice_transpose(uint32_t arr_bitsliced[BITSLICING_BITS_WIDTH]) {
		uint32_t tmp[BITSLICING_BITS_WIDTH];  // arr_bitsliced should also be of this size

		memcpy(tmp, arr_bitsliced, BITSLICING_BITS_WIDTH * sizeof(uint32_t));

		for (int i = 0; i < BITSLICING_BITS_WIDTH; ++i) {
			int idx_of_square_transpose = i % INTS_PER_UNBITSLICED_VALUE;
			int idx_within_square_transpose = i / INTS_PER_UNBITSLICED_VALUE;
			int unbitsliced_origin_of_chunk = 32 * idx_of_square_transpose + idx_within_square_transpose;
			arr_bitsliced[unbitsliced_origin_of_chunk] = tmp[i];
		}

		for (int square_chunk = 0; square_chunk < INTS_PER_UNBITSLICED_VALUE; ++square_chunk) {
			transpose32(arr_bitsliced + 32 * square_chunk);
		}
	}

	__host__ __device__ static void bitslice_untranspose(uint32_t arr_bitsliced[BITSLICING_BITS_WIDTH]) {
		uint32_t tmp[BITSLICING_BITS_WIDTH];  // arr_bitsliced should also be of this size

		memcpy(tmp, arr_bitsliced, BITSLICING_BITS_WIDTH * sizeof(uint32_t));

		for (int square_chunk = 0; square_chunk < INTS_PER_UNBITSLICED_VALUE; ++square_chunk) {
			transpose32(tmp + 32 * square_chunk);
		}

		for (int i = 0; i < BITSLICING_BITS_WIDTH; ++i) {
			int chunk_of_number_idx = i / 32;
			int number_idx = i % 32;
			int unbitsliced_destination_of_chunk = INTS_PER_UNBITSLICED_VALUE * number_idx + chunk_of_number_idx;
			arr_bitsliced[unbitsliced_destination_of_chunk] = tmp[i];
		}
	}

	static void repeat_value_bitsliced(
		uint32_t batch[BITSLICING_BITS_WIDTH], const uint32_t value[INTS_PER_UNBITSLICED_VALUE]
	) {
		for (int i = 0; i < BITSLICING_BITS_WIDTH; ++i) {
			batch[i] = value[i % INTS_PER_UNBITSLICED_VALUE];
		}

		bitslice_transpose(batch);
	}

	__host__ __device__ void bitslice_transpose() { bitslice_transpose(buffer.data()); }

	__host__ __device__ void bitslice_untranspose() { bitslice_transpose(buffer.data()); }

	__host__ __device__ void repeat_value_bitsliced(const std::array<uint32_t, INTS_PER_UNBITSLICED_VALUE>& value) {
		for (int i = 0; i < BITSLICING_BITS_WIDTH; ++i) {
			buffer[i] = value[i % INTS_PER_UNBITSLICED_VALUE];
		}

		bitslice_transpose();
	}
};

template <size_t BITSLICING_BITS_WIDTH>
__global__ static void transpose_kernel(uint32_t* buffer, const uint32_t batches_in_buffer) {
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (uint32_t batch_idx = tid; batch_idx < batches_in_buffer; batch_idx += gridDim.x * blockDim.x) {
		BitsliceUtils<BITSLICING_BITS_WIDTH>::bitslice_transpose(buffer + BITSLICING_BITS_WIDTH * batch_idx);
	}
}

template <size_t BITSLICING_BITS_WIDTH>
__global__ static void untranspose_kernel(uint32_t* buffer, const uint32_t batches_in_buffer) {
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	for (uint32_t batch_idx = tid; batch_idx < batches_in_buffer; batch_idx += gridDim.x * blockDim.x) {
		BitsliceUtils<BITSLICING_BITS_WIDTH>::bitslice_untranspose(buffer + BITSLICING_BITS_WIDTH * batch_idx);
	}
}