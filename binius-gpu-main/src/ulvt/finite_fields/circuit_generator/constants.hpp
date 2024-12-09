#pragma once
#define TOWER_HEIGHT 7
#define BITS_WIDTH (1 << TOWER_HEIGHT)
#define BLANK_MEMORY_SIZE 10000  // should be at least (3^(h+1)-2^(h+1)-2^h)