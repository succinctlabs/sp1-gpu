#pragma once

#include "../fields/bb31_t.cuh"
#include "../matrix/matrix.cuh"
#include "../utils/runtime.cuh"
#include "alu_base.hpp"
#include "alu_ext.hpp"
#include "batch_fri.hpp"
#include "exp_reverse_bits.hpp"
#include "fri_fold.hpp"
#include "moongate-core-sys-cbindgen.hpp"
#include "poseidon2.hpp"
#include "poseidon2_skinny.hpp"
#include "poseidon2_wide.hpp"
#include "public_values.hpp"
#include "select.hpp"
#include "sp1-core-machine-sys-cbindgen.hpp"
#include "sp1-recursion-core-sys-cbindgen.hpp"

using namespace moongate;

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

        const T* arr = std::bit_cast<T*>(&cols);
        size_t start = (i % 4) * COLUMNS;
        for (size_t j = 0; j < COLUMNS; ++j) {
            trace.values[(i / 4) + (j + start) * trace.height] = arr[j];
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
    recursion_base_alu_generate_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            events,
            nb_events
        );

    return CUDA_SUCCESS_MOON;
}

template<class T>
__global__ void recursion_ext_alu_generate_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_recursion_core_sys::ExtAluEvent<T>* events,
    uintptr_t nb_events
) {
    static const size_t COLUMNS =
        sizeof(sp1_recursion_core_sys::ExtAluValueCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x) {
        sp1_recursion_core_sys::ExtAluValueCols<T> cols;
        sp1_recursion_core_sys::alu_ext::event_to_row<T>(events[i], cols);

        const T* arr = std::bit_cast<T*>(&cols);
        size_t start = (i % 4) * COLUMNS;
        for (size_t j = 0; j < COLUMNS; ++j) {
            trace.values[(i / 4) + (j + start) * trace.height] = arr[j];
        }
    }
}

extern "C" rustCudaError_t recursion_ext_alu_generate_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_recursion_core_sys::ExtAluEvent<bb31_t>* events,
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
    recursion_ext_alu_generate_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            events,
            nb_events
        );

    return CUDA_SUCCESS_MOON;
}

template<class T>
__global__ void recursion_batch_fri_generate_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_recursion_core_sys::BatchFRIEvent<T>* events,
    uintptr_t nb_events
) {
    static const size_t COLUMNS =
        sizeof(sp1_recursion_core_sys::BatchFRICols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x) {
        sp1_recursion_core_sys::BatchFRICols<T> cols;
        sp1_recursion_core_sys::batch_fri::event_to_row<T>(events[i], cols);

        const T* arr = std::bit_cast<T*>(&cols);
        for (size_t j = 0; j < COLUMNS; ++j) {
            trace.values[i + j * trace.height] = arr[j];
        }
    }
}

extern "C" rustCudaError_t recursion_batch_fri_generate_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_recursion_core_sys::BatchFRIEvent<bb31_t>* events,
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
    recursion_batch_fri_generate_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            events,
            nb_events
        );

    return CUDA_SUCCESS_MOON;
}

template<class T>
__global__ void recursion_exp_reverse_bits_generate_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_recursion_core_sys::ExpReverseBitsEvent<T>* events,
    uintptr_t nb_events
) {
    static const size_t COLUMNS =
        sizeof(sp1_recursion_core_sys::ExpReverseBitsLenCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x) {
        // Per event accumulator
        T accum = T::one();

        for (size_t exp_idx = 0; exp_idx < events[i].len; ++exp_idx) {
            sp1_recursion_core_sys::ExpReverseBitsLenCols<T> cols;
            sp1_recursion_core_sys::exp_reverse_bits::event_to_row<T>(
                events[i],
                exp_idx,
                cols
            );

            T prev_accum = accum;
            accum = prev_accum * prev_accum * cols.multiplier;

            cols.accum = accum;
            cols.accum_squared = accum * accum;
            cols.prev_accum_squared = prev_accum * prev_accum;
            cols.prev_accum_squared_times_multiplier =
                cols.prev_accum_squared * cols.multiplier;

            const T* arr = std::bit_cast<T*>(&cols);
            for (size_t j = 0; j < COLUMNS; ++j) {
                trace.values[i + exp_idx + j * trace.height] = arr[j];
            }
        }
    }
}

extern "C" rustCudaError_t recursion_exp_reverse_bits_generate_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_recursion_core_sys::ExpReverseBitsEvent<bb31_t>* events,
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
    recursion_exp_reverse_bits_generate_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            events,
            nb_events
        );

    return CUDA_SUCCESS_MOON;
}

template<class T>
__global__ void recursion_fri_fold_generate_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_recursion_core_sys::FriFoldEvent<T>* events,
    uintptr_t nb_events
) {
    static const size_t COLUMNS =
        sizeof(sp1_recursion_core_sys::FriFoldCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x) {
        sp1_recursion_core_sys::FriFoldCols<T> cols;
        sp1_recursion_core_sys::fri_fold::event_to_row<T>(events[i], cols);

        const T* arr = std::bit_cast<T*>(&cols);
        for (size_t j = 0; j < COLUMNS; ++j) {
            trace.values[i + j * trace.height] = arr[j];
        }
    }
}

extern "C" rustCudaError_t recursion_fri_fold_generate_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_recursion_core_sys::FriFoldEvent<bb31_t>* events,
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
    recursion_fri_fold_generate_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            events,
            nb_events
        );

    return CUDA_SUCCESS_MOON;
}

template<class T>
__global__ void recursion_public_values_generate_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_recursion_core_sys::CommitPublicValuesEvent<T>* events,
    uintptr_t nb_events
) {
    static const size_t COLUMNS =
        sizeof(sp1_recursion_core_sys::PublicValuesCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x) {
        for (size_t digest_idx = 0;
             digest_idx < sp1_recursion_core_sys::DIGEST_SIZE;
             ++digest_idx) {
            sp1_recursion_core_sys::PublicValuesCols<T> cols;

            sp1_recursion_core_sys::public_values::event_to_row<T>(
                events[i],
                digest_idx,
                cols
            );

            const T* arr = std::bit_cast<T*>(&cols);
            for (size_t j = 0; j < COLUMNS; ++j) {
                trace.values[i + digest_idx + j * trace.height] = arr[j];
            }
        }
    }
}

extern "C" rustCudaError_t recursion_public_values_generate_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_recursion_core_sys::CommitPublicValuesEvent<bb31_t>* events,
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
    recursion_public_values_generate_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            events,
            nb_events
        );

    return CUDA_SUCCESS_MOON;
}

template<class T>
__global__ void recursion_select_generate_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_recursion_core_sys::SelectEvent<T>* events,
    uintptr_t nb_events
) {
    static const size_t COLUMNS =
        sizeof(sp1_recursion_core_sys::SelectCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x) {
        sp1_recursion_core_sys::SelectCols<T> cols;
        sp1_recursion_core_sys::select::event_to_row<T>(events[i], cols);

        const T* arr = std::bit_cast<T*>(&cols);
        for (size_t j = 0; j < COLUMNS; ++j) {
            trace.values[i + j * trace.height] = arr[j];
        }
    }
}

extern "C" rustCudaError_t recursion_select_generate_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_recursion_core_sys::SelectEvent<bb31_t>* events,
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
    recursion_select_generate_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            events,
            nb_events
        );

    return CUDA_SUCCESS_MOON;
}

template<class T>
__global__ void recursion_poseidon2_skinny_generate_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_recursion_core_sys::Poseidon2Event<T>* events,
    uintptr_t nb_events
) {
    static const size_t COLUMNS =
        sizeof(sp1_recursion_core_sys::Poseidon2<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x) {
        sp1_recursion_core_sys::Poseidon2<T>
            cols[sp1_recursion_core_sys::OUTPUT_ROUND_IDX + 1];
        sp1_recursion_core_sys::poseidon2_skinny::event_to_row<T>(
            events[i],
            cols
        );

        size_t base_row = i * (sp1_recursion_core_sys::OUTPUT_ROUND_IDX + 1);
        for (size_t round_idx = 0;
             round_idx < (sp1_recursion_core_sys::OUTPUT_ROUND_IDX + 1);
             ++round_idx) {
            const T* arr = std::bit_cast<T*>(&cols[round_idx]);
            size_t row = base_row + round_idx;

            for (size_t j = 0; j < COLUMNS; ++j) {
                trace.values[row + j * trace.height] = arr[j];
            }
        }
    }
}

extern "C" rustCudaError_t recursion_poseidon2_skinny_generate_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_recursion_core_sys::Poseidon2Event<bb31_t>* events,
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

    CUDA_OK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

    static const int M = 256;
    recursion_poseidon2_skinny_generate_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            events,
            nb_events
        );

    return CUDA_SUCCESS_MOON;
}

template<class T>
__global__ void recursion_poseidon2_wide_generate_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_recursion_core_sys::Poseidon2Event<T>* events,
    uintptr_t nb_events
) {
    static const size_t COLUMNS =
        sizeof(sp1_recursion_core_sys::Poseidon2<T>) / sizeof(T);

    bool sbox_state =
        trace.width == sp1_recursion_core_sys::poseidon2::PERMUTATION_SBOX;
    T dummy_input[WIDTH];
    for (size_t i = 0; i < WIDTH; ++i) {
        dummy_input[i] = T::zero();
    }

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < trace.height; i += blockDim.x * gridDim.x) {
        if (i < nb_events) {
            sp1_recursion_core_sys::poseidon2_wide::event_to_row<T>(
                events[i].input,
                trace.values,
                i,
                trace.height,
                sbox_state
            );
        } else {
            sp1_recursion_core_sys::poseidon2_wide::event_to_row<T>(
                dummy_input,
                trace.values,
                i,
                trace.height,
                sbox_state
            );
        }
    }
}

extern "C" rustCudaError_t recursion_poseidon2_wide_generate_trace(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_recursion_core_sys::Poseidon2Event<bb31_t>* events,
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

    CUDA_OK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    static const int M = 256;
    recursion_poseidon2_wide_generate_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            events,
            nb_events
        );

    return CUDA_SUCCESS_MOON;
}
