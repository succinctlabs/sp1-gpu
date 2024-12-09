#pragma once
#include <cstdint>

__global__ void bitsliced_repeat(uint32_t* x, uint32_t* y, uint32_t* dst);