#pragma once

template<typename T>
struct Matrix {
    T* values;
    size_t width;
    size_t height;
    bool row_major;
};
