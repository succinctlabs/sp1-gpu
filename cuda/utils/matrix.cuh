#pragma once

#include "../fields/bb31_t.cuh"
#include <index.h>
#include <equations.h>
#include <enact.cu>

struct RowMajorMatrix {
    bb31_t *values;
    size_t width;
    size_t height;
};


extern "C" void transpose_inplace(RowMajorMatrix *matrix) {
    inplace::transpose_fn(true, matrix->values, matrix->height, matrix->width);
}

