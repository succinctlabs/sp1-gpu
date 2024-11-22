#pragma once

#include "../fields/bb31_t.cuh"
#include "../matrix/matrix.cuh"
#include "../utils/runtime.cuh"
#include "add_sub.hpp"
#include "moongate-core-sys-cbindgen.hpp"
#include "sp1-core-machine-sys-cbindgen.hpp"
#include "sp1-recursion-core-sys-cbindgen.hpp"
#include "alu_base.hpp"

using namespace moongate;

template<class T>
__global__ void core_add_sub_generate_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_core_machine_sys::AluEvent* events,
    uintptr_t nb_events
) {
    static const size_t COLUMNS =
        sizeof(sp1_core_machine_sys::AddSubCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x) {
        sp1_core_machine_sys::AddSubCols<T> cols;
        sp1_core_machine_sys::add_sub::event_to_row<T>(events[i], cols);

        const T* arr = reinterpret_cast<T*>(&cols);
        for (size_t j = 0; j < COLUMNS; ++j) {
            trace.values[i + j * trace.height] = arr[j];
        }
    }
}

extern "C" rustCudaError_t core_add_sub_generate_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_core_machine_sys::AluEvent* events,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    CUDA_OK(cudaMemsetAsync(
        trace.values,
        0,
        trace.width * trace.height * sizeof(bb31_t),
        stream
    ));

    static const int M = 256;
    core_add_sub_generate_trace_kernel<bb31_t><<<(trace.height - 1) / M + 1, M, 0, stream>>>(
        trace,
        events,
        nb_events
    );

    return CUDA_SUCCESS_MOON;
}

template<class T>
__global__ void recursion_base_alu_generate_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_recursion_core_sys::BaseAluEvent<T>* events,
    uintptr_t nb_events
) {
    static const size_t COLUMNS =
        sizeof(sp1_recursion_core_sys::BaseAluValueCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x) {
        sp1_recursion_core_sys::BaseAluValueCols<T> cols;
        sp1_recursion_core_sys::alu_base::event_to_row<T>(events[i], cols);

        const T* arr = reinterpret_cast<T*>(&cols);
        for (size_t j = 0; j < COLUMNS; ++j) {
            trace.values[i + j * trace.height] = arr[j];
        }
    }
}

extern "C" rustCudaError_t recursion_base_alu_generate_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_recursion_core_sys::BaseAluEvent<bb31_t>* events,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    CUDA_OK(cudaMemsetAsync(
        trace.values,
        0,
        trace.width * trace.height * sizeof(bb31_t),
        stream
    ));

    static const int M = 256;
    recursion_base_alu_generate_trace_kernel<bb31_t><<<(trace.height - 1) / M + 1, M, 0, stream>>>(
        trace,
        events,
        nb_events
    );

    return CUDA_SUCCESS_MOON;
}  // namespace moongate::recursion::base_alu
