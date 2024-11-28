#pragma once

#include "../fields/bb31_t.cuh"
#include "../matrix/matrix.cuh"
#include "../utils/runtime.cuh"
#include "sp1-core-machine-sys-cbindgen.hpp"
#include "sp1-recursion-core-sys-cbindgen.hpp"

using namespace moongate;

template<class T>
__global__ void recursion_base_alu_generate_preprocessed_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_recursion_core_sys::BaseAluInstr<T>* instructions,
    uintptr_t nb_instructions
) {
    static const size_t COLUMNS =
        sizeof(sp1_recursion_core_sys::BaseAluAccessCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_instructions; i += blockDim.x * gridDim.x) {
        sp1_recursion_core_sys::BaseAluAccessCols<T> cols;
        sp1_recursion_core_sys::alu_base::instr_to_row<T>(
            instructions[i],
            cols
        );

        const T* arr = reinterpret_cast<T*>(&cols);
        size_t start = (i % 4) * COLUMNS;
        for (size_t j = 0; j < COLUMNS; ++j) {
            trace.values[(i / 4) + (j + start) * trace.height] = arr[j];
        }
    }
}

extern "C" rustCudaError_t recursion_base_alu_generate_preprocessed_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_recursion_core_sys::BaseAluInstr<bb31_t>* instructions,
    uintptr_t nb_instructions,
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
    recursion_base_alu_generate_preprocessed_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            instructions,
            nb_instructions
        );

    return CUDA_SUCCESS_MOON;
}

template<class T>
__global__ void recursion_ext_alu_generate_preprocessed_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_recursion_core_sys::ExtAluInstr<T>* instructions,
    uintptr_t nb_instructions
) {
    static const size_t COLUMNS =
        sizeof(sp1_recursion_core_sys::ExtAluAccessCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_instructions; i += blockDim.x * gridDim.x) {
        sp1_recursion_core_sys::ExtAluAccessCols<T> cols;
        sp1_recursion_core_sys::alu_ext::instr_to_row<T>(instructions[i], cols);

        const T* arr = reinterpret_cast<T*>(&cols);
        size_t start = (i % 4) * COLUMNS;
        for (size_t j = 0; j < COLUMNS; ++j) {
            trace.values[(i / 4) + (j + start) * trace.height] = arr[j];
        }
    }
}

extern "C" rustCudaError_t recursion_ext_alu_generate_preprocessed_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_recursion_core_sys::ExtAluInstr<bb31_t>* instructions,
    uintptr_t nb_instructions,
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
    recursion_ext_alu_generate_preprocessed_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            instructions,
            nb_instructions
        );

    return CUDA_SUCCESS_MOON;
}

template<class T>
__global__ void recursion_public_values_generate_preprocessed_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_recursion_core_sys::CommitPublicValuesInstr<T>* instructions,
    uintptr_t nb_instructions
) {
    static const size_t COLUMNS =
        sizeof(sp1_recursion_core_sys::PublicValuesPreprocessedCols<T>)
        / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_instructions; i += blockDim.x * gridDim.x) {
        for (size_t digest_idx = 0;
             digest_idx < sp1_recursion_core_sys::DIGEST_SIZE;
             ++digest_idx) {
            sp1_recursion_core_sys::PublicValuesPreprocessedCols<T> cols;

            sp1_recursion_core_sys::public_values::instr_to_row<T>(
                instructions[i],
                digest_idx,
                cols
            );

            const T* arr = reinterpret_cast<T*>(&cols);
            for (size_t j = 0; j < COLUMNS; ++j) {
                trace.values[i + digest_idx + j * trace.height] = arr[j];
            }
        }
    }
}

extern "C" rustCudaError_t recursion_public_values_generate_preprocessed_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_recursion_core_sys::CommitPublicValuesInstr<bb31_t>* instructions,
    uintptr_t nb_instructions,
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
    recursion_public_values_generate_preprocessed_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            instructions,
            nb_instructions
        );

    return CUDA_SUCCESS_MOON;
}

template<class T>
__global__ void recursion_select_generate_preprocessed_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_recursion_core_sys::SelectInstr<T>* instructions,
    uintptr_t nb_instructions
) {
    static const size_t COLUMNS =
        sizeof(sp1_recursion_core_sys::SelectPreprocessedCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_instructions; i += blockDim.x * gridDim.x) {
        sp1_recursion_core_sys::SelectPreprocessedCols<T> cols;
        sp1_recursion_core_sys::select::instr_to_row<T>(instructions[i], cols);

        const T* arr = reinterpret_cast<T*>(&cols);
        for (size_t j = 0; j < COLUMNS; ++j) {
            trace.values[i + j * trace.height] = arr[j];
        }
    }
}

extern "C" rustCudaError_t recursion_select_generate_preprocessed_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_recursion_core_sys::SelectInstr<bb31_t>* instructions,
    uintptr_t nb_instructions,
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
    recursion_select_generate_preprocessed_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            instructions,
            nb_instructions
        );

    return CUDA_SUCCESS_MOON;
}