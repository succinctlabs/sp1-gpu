

#ifndef __ROW_MAJOR_MATRIX__
#define __ROW_MAJOR_MATRIX__

#include "../fields/bb31_t.cuh"

struct RowMajorMatrix {
    bb31_t *values;
    size_t width;
    size_t height;
};


#endif;