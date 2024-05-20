#pragma once

#include "../fields/bb31_t.cuh"

struct Matrix {
    bb31_t *values;
    size_t width;
    size_t height;
    bool row_major;
};



