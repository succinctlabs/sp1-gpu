#include <array>
#include <iostream>

#include "../sumcheck.cuh"

struct Benchmarks {
	double memcpy;
	double transpose;
	double raw;
};

template <uint32_t NUM_VARS, uint32_t COMPOSITION_SIZE>
Benchmarks benchmark_one_sample() {
	constexpr uint32_t EVALS_PER_MULTILINEAR = 1 << NUM_VARS;
	constexpr uint32_t INTERPOLATION_POINTS = COMPOSITION_SIZE + 1;
	const size_t total_ints = INTS_PER_VALUE * EVALS_PER_MULTILINEAR * COMPOSITION_SIZE;
	std::vector<uint32_t> multilinear_evals(total_ints);

	Sumcheck<NUM_VARS, COMPOSITION_SIZE, false> s(multilinear_evals, true);

	for (uint32_t round = 0; round < NUM_VARS; ++round) {
		std::array<uint32_t, INTS_PER_VALUE> sum;
		std::array<uint32_t, INTERPOLATION_POINTS * INTS_PER_VALUE> points;

		s.this_round_messages(sum, points);

		std::array<uint32_t, INTS_PER_VALUE> challenge;

		s.move_to_next_round(challenge);
	}

	std::array<uint32_t, INTS_PER_VALUE> sum;
	std::array<uint32_t, INTERPOLATION_POINTS * INTS_PER_VALUE> points;

	s.this_round_messages(sum, points);

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> memcpy = s.start_before_transpose - s.start_before_memcpy;

	std::chrono::duration<double, std::milli> transpose = s.start_raw - s.start_before_transpose;

	std::chrono::duration<double, std::milli> raw = end - s.start_raw;

	return Benchmarks{memcpy.count(), transpose.count(), raw.count()};
}

template <uint32_t NUM_VARS, uint32_t COMPOSITION_SIZE>
void benchmark(int num_runs) {
	std::cout << "NUM_VARS: " << NUM_VARS << " COMPOSITION_SIZE: " << COMPOSITION_SIZE << std::endl;

	benchmark_one_sample<NUM_VARS, COMPOSITION_SIZE>();

	double total_memcpy = 0;
	double total_transpose = 0;
	double total_raw = 0;

	for (int i = 0; i < num_runs; ++i) {
		Benchmarks bench_results = benchmark_one_sample<NUM_VARS, COMPOSITION_SIZE>();
		total_memcpy += bench_results.memcpy;
		total_transpose += bench_results.transpose;
		total_raw += bench_results.raw;
	}

	std::cout << "Memcpy: " << total_memcpy / num_runs << std::endl;
	std::cout << "Transpose: " << total_transpose / num_runs << std::endl;
	std::cout << "Raw: " << total_raw / num_runs << std::endl;
	std::cout << "Total: " << (total_memcpy + total_transpose + total_raw) / num_runs << std::endl;
}

int main() {
	benchmark<20, 2>(10);
	benchmark<20, 3>(10);
	benchmark<20, 4>(10);

	benchmark<24, 2>(5);
	benchmark<24, 3>(5);
	benchmark<24, 4>(5);

	benchmark<28, 2>(2);
	benchmark<28, 3>(2);
	benchmark<28, 4>(2);

	return 0;
}