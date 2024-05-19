#include <stdexcept>

#ifndef __UTILS_VEC_MATRIX__
#define __UTILS_VEC_MATRIX__

template <typename T>
struct DeviceSlice {
    T *items;
    size_t length;

    DeviceSlice() : items(nullptr), length(0) {}

    __host__ __device__ DeviceSlice(T *items, size_t length) {
        this->items = items;
        this->length = length;
    }

    __device__ T &operator[](size_t index) { return items[index]; }

    __device__ const T &operator[](size_t index) const { return items[index]; }

    __device__ DeviceSlice<T> slice(size_t offset, size_t length) const {
        return DeviceSlice<T>(items + offset, length);
    }
};

#endif
