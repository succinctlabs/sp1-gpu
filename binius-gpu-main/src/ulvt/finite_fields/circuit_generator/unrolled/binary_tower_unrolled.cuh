#pragma once
#include <cstdint>

template <int HEIGHT>
__host__ __device__ void multiply_unrolled(const uint32_t *field_element_a, const uint32_t *field_element_b, uint32_t *destination);
