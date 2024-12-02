#pragma once

#include "../fields/bb31_curve_t.cuh"
#include "../fields/bb31_t.cuh"
#include "../matrix/matrix.cuh"
#include "../matrix/transpose.cuh"
#include "../utils/runtime.cuh"
#include "add_sub.hpp"
#include "alu_base.hpp"
#include "memory_local.hpp"
#include "memory_global.hpp"
#include "syscall.hpp"
#include "moongate-core-sys-cbindgen.hpp"
#include "recursion.cuh"
#include "sp1-core-machine-sys-cbindgen.hpp"
#include "sp1-recursion-core-sys-cbindgen.hpp"

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
    core_add_sub_generate_trace_kernel<bb31_t>
        <<<(trace.height - 1) / M + 1, M, 0, stream>>>(
            trace,
            events,
            nb_events
        );

    return CUDA_SUCCESS_MOON;
}

// MemoryLocalChip
template<class F, class EF7>
__global__ void core_memory_local_generate_trace_decompress_kernel(
    MatrixViewMutDevice<F> trace,
    const sp1_core_machine_sys::MemoryLocalEvent* events,
    uintptr_t nb_events
) {
    // static const size_t MEMORY_LOCAL_COLUMNS =
    //     sizeof(sp1_core_machine_sys::MemoryLocalCols<F>) / sizeof(F);

    // static const size_t SINGLE_MEMORY_LOCAL_COLUMNS =
    //     sizeof(sp1_core_machine_sys::SingleMemoryLocal<F>) / sizeof(F);

    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // #pragma unroll(1)
    // for (; i < trace.height; i += blockDim.x * gridDim.x) {
    //     // ok so we're on the ith row
    //     bb31_septic_curve_t sum = bb31_septic_curve_t();
    //     if (i == 0) {
    //         sum = bb31_septic_curve_t::start_point();
    //     }
    //     sp1_core_machine_sys::MemoryLocalCols<F> cols;
    //     F* cols_arr = reinterpret_cast<F*>(&cols);
    //     for (int k = 0; k < MEMORY_LOCAL_COLUMNS; k++) {
    //         cols_arr[k] = F::zero();
    //     }
    //     for (int j = 0; j < 4; j++) {
    //         int event_idx = 4 * i + j;
    //         if (event_idx < nb_events) {
    //             sp1_core_machine_sys::memory_local::event_to_row<F, EF7>(
    //                 &events[event_idx],
    //                 &cols.memory_local_entries[j]
    //             );
    //             {
    //                 cols.global_accumulation_cols.cumulative_sum[2 * j][0] =
    //                     cols.memory_local_entries[j]
    //                         .initial_global_interaction_cols.x_coordinate;
    //                 cols.global_accumulation_cols.cumulative_sum[2 * j][1] =
    //                     cols.memory_local_entries[j]
    //                         .initial_global_interaction_cols.y_coordinate;
    //                 bb31_septic_curve_t point = bb31_septic_curve_t(
    //                     cols.memory_local_entries[j]
    //                         .initial_global_interaction_cols.x_coordinate._0,
    //                     cols.memory_local_entries[j]
    //                         .initial_global_interaction_cols.y_coordinate._0
    //                 );
    //                 sum += point;
    //             }
    //             {
    //                 cols.global_accumulation_cols.cumulative_sum[2 * j + 1][0] =
    //                     cols.memory_local_entries[j]
    //                         .final_global_interaction_cols.x_coordinate;
    //                 cols.global_accumulation_cols.cumulative_sum[2 * j + 1][1] =
    //                     cols.memory_local_entries[j]
    //                         .final_global_interaction_cols.y_coordinate;
    //                 bb31_septic_curve_t point = bb31_septic_curve_t(
    //                     cols.memory_local_entries[j]
    //                         .final_global_interaction_cols.x_coordinate._0,
    //                     cols.memory_local_entries[j]
    //                         .final_global_interaction_cols.y_coordinate._0
    //                 );
    //                 sum += point;
    //             }
    //         }
    //     }
    //     for (int k = 0; k < 7; k++) {
    //         cols.global_accumulation_cols.initial_digest[0]._0[k] =
    //             sum.x.value[k];
    //         cols.global_accumulation_cols.initial_digest[1]._0[k] =
    //             sum.y.value[k];
    //     }
    //     const F* arr = reinterpret_cast<F*>(&cols);
    //     for (size_t k = 0; k < MEMORY_LOCAL_COLUMNS; ++k) {
    //         trace.values[i + k * trace.height] = arr[k];
    //     }
    // }
}

template<class F, class EF7>
__global__ void core_memory_local_generate_trace_finalize_kernel(
    MatrixViewMutDevice<F> trace,
    bb31_septic_curve_t* cumulative_sums,
    uintptr_t nb_events
) {
//     static const size_t MEMORY_LOCAL_COLUMNS =
//         sizeof(sp1_core_machine_sys::MemoryLocalCols<F>) / sizeof(F);

//     static const size_t SINGLE_MEMORY_LOCAL_COLUMNS =
//         sizeof(sp1_core_machine_sys::SingleMemoryLocal<F>) / sizeof(F);

//     int i = blockIdx.x * blockDim.x + threadIdx.x;

// #pragma unroll(1)
//     for (; i < trace.height; i += blockDim.x * gridDim.x) {
//         sp1_core_machine_sys::MemoryLocalCols<F> cols;
//         F* temp_arr = reinterpret_cast<F*>(&cols);
//         for (int j = 0; j < MEMORY_LOCAL_COLUMNS; j++) {
//             temp_arr[j] = trace.values[i + j * trace.height];
//         }

//         bb31_septic_curve_t sum = cumulative_sums[i];

//         for (int j = 3; j >= 0; j--) {
//             int event_idx = 4 * i + j;
//             {
//                 {
//                     bb31_septic_extension_t point_x = bb31_septic_extension_t(
//                         cols.memory_local_entries[j]
//                             .final_global_interaction_cols.x_coordinate._0
//                     );
//                     bb31_septic_extension_t point_y =
//                         bb31_septic_extension_t(
//                             cols.memory_local_entries[j]
//                                 .final_global_interaction_cols.y_coordinate._0
//                         )
//                         * (bb31_t::zero() - bb31_t::one());
//                     bb31_septic_curve_t point =
//                         bb31_septic_curve_t(point_x, point_y);
//                     for (int k = 0; k < 7; k++) {
//                         cols.global_accumulation_cols
//                             .cumulative_sum[2 * j + 1][0]
//                             ._0[k] = sum.x.value[k];
//                         cols.global_accumulation_cols
//                             .cumulative_sum[2 * j + 1][1]
//                             ._0[k] = sum.y.value[k];
//                     }
//                     sum += point;
//                     if (event_idx >= nb_events) {
//                         bb31_septic_curve_t dummy =
//                             bb31_septic_curve_t::dummy_point();
//                         for (int k = 0; k < 7; k++) {
//                             cols.memory_local_entries[j]
//                                 .final_global_interaction_cols.x_coordinate
//                                 ._0[k] = dummy.x.value[k];
//                             cols.memory_local_entries[j]
//                                 .final_global_interaction_cols.y_coordinate
//                                 ._0[k] = dummy.y.value[k];
//                         }
//                     }
//                 }

//                 {
//                     bb31_septic_extension_t point_x = bb31_septic_extension_t(
//                         cols.memory_local_entries[j]
//                             .initial_global_interaction_cols.x_coordinate._0
//                     );
//                     bb31_septic_extension_t point_y =
//                         bb31_septic_extension_t(
//                             cols.memory_local_entries[j]
//                                 .initial_global_interaction_cols.y_coordinate._0
//                         )
//                         * (bb31_t::zero() - bb31_t::one());
//                     bb31_septic_curve_t point =
//                         bb31_septic_curve_t(point_x, point_y);
//                     for (int k = 0; k < 7; k++) {
//                         cols.global_accumulation_cols.cumulative_sum[2 * j][0]
//                             ._0[k] = sum.x.value[k];
//                         cols.global_accumulation_cols.cumulative_sum[2 * j][1]
//                             ._0[k] = sum.y.value[k];
//                     }
//                     sum += point;
//                     if (event_idx >= nb_events) {
//                         bb31_septic_curve_t dummy =
//                             bb31_septic_curve_t::dummy_point();
//                         for (int k = 0; k < 7; k++) {
//                             cols.memory_local_entries[j]
//                                 .initial_global_interaction_cols.x_coordinate
//                                 ._0[k] = dummy.x.value[k];
//                             cols.memory_local_entries[j]
//                                 .initial_global_interaction_cols.y_coordinate
//                                 ._0[k] = dummy.y.value[k];
//                         }
//                     }
//                 }
//             }
//         }
//         for (int k = 0; k < 7; k++) {
//             cols.global_accumulation_cols.initial_digest[0]._0[k] =
//                 sum.x.value[k];
//             cols.global_accumulation_cols.initial_digest[1]._0[k] =
//                 sum.y.value[k];
//         }
//         for (int j = 0; j < 8; j++) {
//             if (4 * i + j / 2 < nb_events) {
//                 for (int k = 0; k < 7; k++) {
//                     cols.global_accumulation_cols.sum_checker[j]._0[k] =
//                         bb31_t::zero();
//                 }
//             } else {
//                 bb31_septic_curve_t dummy = bb31_septic_curve_t::dummy_point();
//                 bb31_septic_curve_t digest = bb31_septic_curve_t(
//                     cols.global_accumulation_cols.cumulative_sum[j][0]._0,
//                     cols.global_accumulation_cols.cumulative_sum[j][1]._0
//                 );
//                 bb31_septic_extension_t sum_checker_x =
//                     bb31_septic_curve_t::sum_checker_x(digest, dummy, digest);
//                 for (int k = 0; k < 7; k++) {
//                     cols.global_accumulation_cols.sum_checker[j]._0[k] =
//                         sum_checker_x.value[k];
//                 }
//             }
//         }

//         F* final_temp = reinterpret_cast<F*>(&cols);
//         for (int j = 0; j < MEMORY_LOCAL_COLUMNS; j++) {
//             trace.values[i + j * trace.height] = final_temp[j];
//         }
//     }
}

extern "C" rustCudaError_t core_memory_local_generate_trace_round_3(
    MatrixViewMutDevice<bb31_t> trace,  // this is still column major!
    bb31_septic_curve_t* cumulative_sums,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

    static const int M = 64;

    core_memory_local_generate_trace_finalize_kernel<
        bb31_t,
        bb31_septic_extension_t><<<(trace.height - 1) / M + 1, M, 0, stream>>>(
        trace,
        cumulative_sums,
        nb_events
    );
    CUDA_OK(cudaStreamSynchronize(stream));

    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t core_memory_local_generate_trace_round_2(
    MatrixViewMutDevice<bb31_t> trace,
    bb31_septic_curve_t* cumulative_sums,
    CudaStreamHandle stream_handle
) {
    // Get the stream.
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

    // Select the right set of columns from the trace.
    Matrix<bb31_t> initial_digest_trace_col_major;
    initial_digest_trace_col_major.values = trace.values
        + trace.height;
            // * sp1_core_machine_sys::MEMORY_LOCAL_INITIAL_DIGEST_POS_COPY;

    initial_digest_trace_col_major.width = 14;
    initial_digest_trace_col_major.height = trace.height;
    initial_digest_trace_col_major.row_major = false;

    // Allocate memory for the row-major version of the initial digest trace.
    bb31_t* initial_digest_trace_row_major;
    CUDA_OK(cudaMallocAsync(
        &initial_digest_trace_row_major,
        sizeof(bb31_t) * 14 * trace.height,
        stream
    ));

    // Transpose the initial digest trace from column-major to row-major.
    matrix_transpose::transpose_naive(
        initial_digest_trace_row_major,
        initial_digest_trace_col_major,
        stream
    );
    CUDA_OK(cudaStreamSynchronize(stream));

    // Cast the row-major initial digest trace to a curve.
    bb31_septic_curve_t* initial_digest_trace_row_major_curve =
        reinterpret_cast<bb31_septic_curve_t*>(initial_digest_trace_row_major);

    // Compute the cumulative sums of the initial digest trace.
    ScanTemplateLarge(
        cumulative_sums,
        initial_digest_trace_row_major_curve,
        trace.height,
        stream
    );
    CUDA_OK(cudaStreamSynchronize(stream));

    // Free the allocated memory for the row-major initial digest trace.
    CUDA_OK(cudaFreeAsync(initial_digest_trace_row_major, stream));

    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t core_memory_local_generate_trace_round_1(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_core_machine_sys::MemoryLocalEvent* events,
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

    static const int M = 64;

    core_memory_local_generate_trace_decompress_kernel<
        bb31_t,
        bb31_septic_extension_t><<<(trace.height - 1) / M + 1, M, 0, stream>>>(
        trace,
        events,
        nb_events
    );
    CUDA_OK(cudaStreamSynchronize(stream));

    return CUDA_SUCCESS_MOON;
}

// MemoryGlobalChip
template<class F, class EF7>
__global__ void core_memory_global_generate_trace_decompress_kernel(
    MatrixViewMutDevice<F> trace,
    const sp1_core_machine_sys::MemoryInitializeFinalizeEvent* events,
    bool is_receive,
    uint32_t previous_addr,
    uintptr_t nb_events
) {
    // static const size_t MEMORY_INIT_COLUMNS =
    //     sizeof(sp1_core_machine_sys::MemoryInitCols<F>) / sizeof(F);

    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // #pragma unroll(1)
    // for (; i < trace.height; i += blockDim.x * gridDim.x) {
    //     // ok so we're on the ith row
    //     bb31_septic_curve_t sum = bb31_septic_curve_t();
    //     if (i == 0) {
    //         sum = bb31_septic_curve_t::start_point();
    //     }
    //     sp1_core_machine_sys::MemoryInitCols<F> cols;
    //     F* cols_arr = reinterpret_cast<F*>(&cols);
    //     for (int k = 0; k < MEMORY_INIT_COLUMNS; k++) {
    //         cols_arr[k] = F::zero();
    //     }
    //     int event_idx = i;
    //     if (event_idx < nb_events) {
    //         if (i == 0) {
    //             if (previous_addr == 0) {
    //                 cols.is_prev_addr_zero.inverse = F::zero();
    //                 cols.is_prev_addr_zero.result = F::one();
    //                 cols.is_first_comp = F::zero();
    //             } else {
    //                 cols.is_prev_addr_zero.inverse = F::from_canonical_u32(previous_addr).reciprocal();
    //                 cols.is_prev_addr_zero.result = F::zero();
    //                 cols.is_first_comp = F::one();
    //                 for(int idx = 31 ; idx >= 0; idx--) {
    //                     int prev_bit = (previous_addr >> idx) & 1;
    //                     int cur_bit = (events[event_idx].addr >> idx) & 1;
    //                     if (prev_bit == 0 && cur_bit == 1) {
    //                         cols.lt_cols.bit_flags[idx] = F::one();
    //                         break;
    //                     }
    //                 }
    //             }
    //         } else {
    //             cols.is_next_comp = F::from_canonical_u32(events[event_idx - 1].used);
    //             for(int idx = 31 ; idx >= 0; idx--) {
    //                 int prev_bit = (events[event_idx - 1].addr >> idx) & 1;
    //                 int cur_bit = (events[event_idx].addr >> idx) & 1;
    //                 if (prev_bit == 0 && cur_bit == 1) {
    //                     cols.lt_cols.bit_flags[idx] = F::one();
    //                     break;
    //                 }
    //             }
    //         }
    //         sp1_core_machine_sys::memory_global::event_to_row<F, EF7>(
    //             &events[event_idx],
    //             is_receive,
    //             &cols
    //         );
    //         cols.global_accumulation_cols.cumulative_sum[0][0] = 
    //             cols.global_interaction_cols.x_coordinate;
    //         cols.global_accumulation_cols.cumulative_sum[0][1] =
    //             cols.global_interaction_cols.y_coordinate;
    //         bb31_septic_curve_t point = bb31_septic_curve_t(
    //             cols.global_interaction_cols.x_coordinate._0,
    //             cols.global_interaction_cols.y_coordinate._0
    //         );
    //         sum += point;
    //     }
    //     for (int k = 0; k < 7; k++) {
    //         cols.global_accumulation_cols.initial_digest[0]._0[k] =
    //             sum.x.value[k];
    //         cols.global_accumulation_cols.initial_digest[1]._0[k] =
    //             sum.y.value[k];
    //     }
    //     if (nb_events >= 1 && i == nb_events - 1) {
    //         cols.is_last_addr = F::one();
    //     }
    //     const F* arr = reinterpret_cast<F*>(&cols);
    //     for (size_t k = 0; k < MEMORY_INIT_COLUMNS; ++k) {
    //         trace.values[i + k * trace.height] = arr[k];
    //     }
    // }
}

template<class F, class EF7>
__global__ void core_memory_global_generate_trace_finalize_kernel(
    MatrixViewMutDevice<F> trace,
    bb31_septic_curve_t* cumulative_sums,
    uintptr_t nb_events
) {
//     static const size_t MEMORY_INIT_COLUMNS =
//         sizeof(sp1_core_machine_sys::MemoryInitCols<F>) / sizeof(F);

//     int i = blockIdx.x * blockDim.x + threadIdx.x;

// #pragma unroll(1)
//     for (; i < trace.height; i += blockDim.x * gridDim.x) {
//         sp1_core_machine_sys::MemoryInitCols<F> cols;
//         F* temp_arr = reinterpret_cast<F*>(&cols);
//         for (int j = 0; j < MEMORY_INIT_COLUMNS; j++) {
//             temp_arr[j] = trace.values[i + j * trace.height];
//         }

//         bb31_septic_curve_t sum = cumulative_sums[i];

//         int event_idx = i;
            
//         bb31_septic_extension_t point_x = bb31_septic_extension_t(cols.global_interaction_cols.x_coordinate._0);
//         bb31_septic_extension_t point_y = bb31_septic_extension_t(cols.global_interaction_cols.y_coordinate._0) * (bb31_t::zero() - bb31_t::one());
//         bb31_septic_curve_t point = bb31_septic_curve_t(point_x, point_y);
        
//         for (int k = 0; k < 7; k++) {
//             cols.global_accumulation_cols.cumulative_sum[0][0]._0[k] = sum.x.value[k];
//             cols.global_accumulation_cols.cumulative_sum[0][1]._0[k] = sum.y.value[k];
//         }

//         sum += point;

//         for (int k = 0; k < 7; k++) {
//             cols.global_accumulation_cols.initial_digest[0]._0[k] =
//                 sum.x.value[k];
//             cols.global_accumulation_cols.initial_digest[1]._0[k] =
//                 sum.y.value[k];
//         }

//         if (event_idx < nb_events) {
//             for (int k = 0; k < 7; k++) {
//                 cols.global_accumulation_cols.sum_checker[0]._0[k] = F::zero();
//             }
//         } else {
//             bb31_septic_curve_t dummy =
//                 bb31_septic_curve_t::dummy_point();
//             for (int k = 0; k < 7; k++) {
//                 cols.global_interaction_cols.x_coordinate._0[k] = dummy.x.value[k];
//                 cols.global_interaction_cols.y_coordinate._0[k] = dummy.y.value[k];
//             }
//             bb31_septic_curve_t digest = bb31_septic_curve_t(
//                 cols.global_accumulation_cols.cumulative_sum[0][0]._0,
//                 cols.global_accumulation_cols.cumulative_sum[0][1]._0
//             );
//             bb31_septic_extension_t sum_checker_x =
//                 bb31_septic_curve_t::sum_checker_x(digest, dummy, digest);
//             for (int k = 0; k < 7; k++) {
//                 cols.global_accumulation_cols.sum_checker[0]._0[k] =
//                     sum_checker_x.value[k];
//             }
//         }    

//         F* final_temp = reinterpret_cast<F*>(&cols);
//         for (int j = 0; j < MEMORY_INIT_COLUMNS; j++) {
//             trace.values[i + j * trace.height] = final_temp[j];
//         }
//     }
}

extern "C" rustCudaError_t core_memory_global_generate_trace_round_3(
    MatrixViewMutDevice<bb31_t> trace,  // this is still column major!
    bb31_septic_curve_t* cumulative_sums,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

    static const int M = 64;

    core_memory_global_generate_trace_finalize_kernel<
        bb31_t,
        bb31_septic_extension_t><<<(trace.height - 1) / M + 1, M, 0, stream>>>(
        trace,
        cumulative_sums,
        nb_events
    );
    CUDA_OK(cudaStreamSynchronize(stream));

    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t core_memory_global_generate_trace_round_2(
    MatrixViewMutDevice<bb31_t> trace,
    bb31_septic_curve_t* cumulative_sums,
    CudaStreamHandle stream_handle
) {
    // Get the stream.
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

    // Select the right set of columns from the trace.
    Matrix<bb31_t> initial_digest_trace_col_major;
    initial_digest_trace_col_major.values = trace.values
        + trace.height
            * sp1_core_machine_sys::MEMORY_GLOBAL_INITIAL_DIGEST_POS_COPY;
    initial_digest_trace_col_major.width = 14;
    initial_digest_trace_col_major.height = trace.height;
    initial_digest_trace_col_major.row_major = false;

    // Allocate memory for the row-major version of the initial digest trace.
    bb31_t* initial_digest_trace_row_major;
    CUDA_OK(cudaMallocAsync(
        &initial_digest_trace_row_major,
        sizeof(bb31_t) * 14 * trace.height,
        stream
    ));

    // Transpose the initial digest trace from column-major to row-major.
    matrix_transpose::transpose_naive(
        initial_digest_trace_row_major,
        initial_digest_trace_col_major,
        stream
    );
    CUDA_OK(cudaStreamSynchronize(stream));

    // Cast the row-major initial digest trace to a curve.
    bb31_septic_curve_t* initial_digest_trace_row_major_curve =
        reinterpret_cast<bb31_septic_curve_t*>(initial_digest_trace_row_major);

    // Compute the cumulative sums of the initial digest trace.
    ScanTemplateLarge(
        cumulative_sums,
        initial_digest_trace_row_major_curve,
        trace.height,
        stream
    );
    CUDA_OK(cudaStreamSynchronize(stream));

    // Free the allocated memory for the row-major initial digest trace.
    CUDA_OK(cudaFreeAsync(initial_digest_trace_row_major, stream));

    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t core_memory_global_generate_trace_round_1(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_core_machine_sys::MemoryInitializeFinalizeEvent* events,
    uint32_t previous_addr,
    uintptr_t nb_events,
    bool is_receive,
    CudaStreamHandle stream_handle
) {
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    CUDA_OK(cudaMemsetAsync(
        trace.values,
        0,
        trace.width * trace.height * sizeof(bb31_t),
        stream
    ));

    static const int M = 64;

    core_memory_global_generate_trace_decompress_kernel<
        bb31_t,
        bb31_septic_extension_t><<<(trace.height - 1) / M + 1, M, 0, stream>>>(
        trace,
        events,
        is_receive,
        previous_addr,
        nb_events
    );
    CUDA_OK(cudaStreamSynchronize(stream));

    return CUDA_SUCCESS_MOON;
}

// SyscallChip
template<class F, class EF7>
__global__ void core_syscall_generate_trace_decompress_kernel(
    MatrixViewMutDevice<F> trace,
    const sp1_core_machine_sys::SyscallEvent* events,
    bool is_receive,
    uintptr_t nb_events
) {
    // static const size_t SYSCALL_COLUMNS =
    //     sizeof(sp1_core_machine_sys::SyscallCols<F>) / sizeof(F);

    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    // #pragma unroll(1)
    // for (; i < trace.height ; i += blockDim.x * gridDim.x) {
    //     // ok so we're on the ith row
    //     bb31_septic_curve_t sum = bb31_septic_curve_t();
    //     if (i == 0) {
    //         sum = bb31_septic_curve_t::start_point();
    //     }
    //     sp1_core_machine_sys::SyscallCols<F> cols;
    //     F* cols_arr = reinterpret_cast<F*>(&cols);
    //     for (int k = 0; k < SYSCALL_COLUMNS; k++) {
    //         cols_arr[k] = F::zero();
    //     }
    //     int event_idx = i;
    //     if (event_idx < nb_events) {
    //         sp1_core_machine_sys::syscall::event_to_row<F, EF7>(
    //             &events[event_idx],
    //             is_receive,
    //             &cols
    //         );
    //         cols.global_accumulation_cols.cumulative_sum[0][0] = 
    //             cols.global_interaction_cols.x_coordinate;
    //         cols.global_accumulation_cols.cumulative_sum[0][1] =
    //             cols.global_interaction_cols.y_coordinate;
    //         bb31_septic_curve_t point = bb31_septic_curve_t(
    //             cols.global_interaction_cols.x_coordinate._0,
    //             cols.global_interaction_cols.y_coordinate._0
    //         );
    //         sum += point;
    //     }
    //     for (int k = 0; k < 7; k++) {
    //         cols.global_accumulation_cols.initial_digest[0]._0[k] =
    //             sum.x.value[k];
    //         cols.global_accumulation_cols.initial_digest[1]._0[k] =
    //             sum.y.value[k];
    //     }
    //     const F* arr = reinterpret_cast<F*>(&cols);
    //     for (size_t k = 0; k < SYSCALL_COLUMNS ; ++k) {
    //         trace.values[i + k * trace.height] = arr[k];
    //     }
    // }
}

template<class F, class EF7>
__global__ void core_syscall_generate_trace_finalize_kernel(
    MatrixViewMutDevice<F> trace,
    bb31_septic_curve_t* cumulative_sums,
    uintptr_t nb_events
) {
//     static const size_t SYSCALL_COLUMNS =
//         sizeof(sp1_core_machine_sys::SyscallCols<F>) / sizeof(F);

//     int i = blockIdx.x * blockDim.x + threadIdx.x;

// #pragma unroll(1)
//     for (; i < trace.height; i += blockDim.x * gridDim.x) {
//         sp1_core_machine_sys::SyscallCols<F> cols;
//         F* temp_arr = reinterpret_cast<F*>(&cols);
//         for (int j = 0; j < SYSCALL_COLUMNS ; j++) {
//             temp_arr[j] = trace.values[i + j * trace.height];
//         }

//         bb31_septic_curve_t sum = cumulative_sums[i];

//         int event_idx = i;
            
//         bb31_septic_extension_t point_x = bb31_septic_extension_t(cols.global_interaction_cols.x_coordinate._0);
//         bb31_septic_extension_t point_y = bb31_septic_extension_t(cols.global_interaction_cols.y_coordinate._0) * (bb31_t::zero() - bb31_t::one());
//         bb31_septic_curve_t point = bb31_septic_curve_t(point_x, point_y);
        
//         for (int k = 0; k < 7; k++) {
//             cols.global_accumulation_cols.cumulative_sum[0][0]._0[k] = sum.x.value[k];
//             cols.global_accumulation_cols.cumulative_sum[0][1]._0[k] = sum.y.value[k];
//         }

//         sum += point;

//         for (int k = 0; k < 7; k++) {
//             cols.global_accumulation_cols.initial_digest[0]._0[k] =
//                 sum.x.value[k];
//             cols.global_accumulation_cols.initial_digest[1]._0[k] =
//                 sum.y.value[k];
//         }

//         if (event_idx < nb_events) {
//             for (int k = 0; k < 7; k++) {
//                 cols.global_accumulation_cols.sum_checker[0]._0[k] = F::zero();
//             }
//         } else {
//             bb31_septic_curve_t dummy =
//                 bb31_septic_curve_t::dummy_point();
//             for (int k = 0; k < 7; k++) {
//                 cols.global_interaction_cols.x_coordinate._0[k] = dummy.x.value[k];
//                 cols.global_interaction_cols.y_coordinate._0[k] = dummy.y.value[k];
//             }
//             bb31_septic_curve_t digest = bb31_septic_curve_t(
//                 cols.global_accumulation_cols.cumulative_sum[0][0]._0,
//                 cols.global_accumulation_cols.cumulative_sum[0][1]._0
//             );
//             bb31_septic_extension_t sum_checker_x =
//                 bb31_septic_curve_t::sum_checker_x(digest, dummy, digest);
//             for (int k = 0; k < 7; k++) {
//                 cols.global_accumulation_cols.sum_checker[0]._0[k] =
//                     sum_checker_x.value[k];
//             }
//         }    

//         F* final_temp = reinterpret_cast<F*>(&cols);
//         for (int j = 0; j < SYSCALL_COLUMNS ; j++) {
//             trace.values[i + j * trace.height] = final_temp[j];
//         }
//     }
}

extern "C" rustCudaError_t core_syscall_generate_trace_round_3(
    MatrixViewMutDevice<bb31_t> trace,  // this is still column major!
    bb31_septic_curve_t* cumulative_sums,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

    static const int M = 64;

    core_syscall_generate_trace_finalize_kernel<
        bb31_t,
        bb31_septic_extension_t><<<(trace.height - 1) / M + 1, M, 0, stream>>>(
        trace,
        cumulative_sums,
        nb_events
    );
    CUDA_OK(cudaStreamSynchronize(stream));

    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t core_syscall_generate_trace_round_2(
    MatrixViewMutDevice<bb31_t> trace,
    bb31_septic_curve_t* cumulative_sums,
    CudaStreamHandle stream_handle
) {
    // Get the stream.
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

    // Select the right set of columns from the trace.
    Matrix<bb31_t> initial_digest_trace_col_major;
    initial_digest_trace_col_major.values = trace.values
        + trace.height
            * sp1_core_machine_sys::SYSCALL_INITIAL_DIGEST_POS_COPY;
    initial_digest_trace_col_major.width = 14;
    initial_digest_trace_col_major.height = trace.height;
    initial_digest_trace_col_major.row_major = false;

    // Allocate memory for the row-major version of the initial digest trace.
    bb31_t* initial_digest_trace_row_major;
    CUDA_OK(cudaMallocAsync(
        &initial_digest_trace_row_major,
        sizeof(bb31_t) * 14 * trace.height,
        stream
    ));

    // Transpose the initial digest trace from column-major to row-major.
    matrix_transpose::transpose_naive(
        initial_digest_trace_row_major,
        initial_digest_trace_col_major,
        stream
    );
    CUDA_OK(cudaStreamSynchronize(stream));

    // Cast the row-major initial digest trace to a curve.
    bb31_septic_curve_t* initial_digest_trace_row_major_curve =
        reinterpret_cast<bb31_septic_curve_t*>(initial_digest_trace_row_major);

    // Compute the cumulative sums of the initial digest trace.
    ScanTemplateLarge(
        cumulative_sums,
        initial_digest_trace_row_major_curve,
        trace.height,
        stream
    );
    CUDA_OK(cudaStreamSynchronize(stream));

    // Free the allocated memory for the row-major initial digest trace.
    CUDA_OK(cudaFreeAsync(initial_digest_trace_row_major, stream));

    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t core_syscall_generate_trace_round_1(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_core_machine_sys::SyscallEvent* events,
    uintptr_t nb_events,
    bool is_receive,
    CudaStreamHandle stream_handle
) {
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    CUDA_OK(cudaMemsetAsync(
        trace.values,
        0,
        trace.width * trace.height * sizeof(bb31_t),
        stream
    ));

    static const int M = 64;

    core_syscall_generate_trace_decompress_kernel<
        bb31_t,
        bb31_septic_extension_t><<<(trace.height - 1) / M + 1, M, 0, stream>>>(
        trace,
        events,
        is_receive,
        nb_events
    );
    CUDA_OK(cudaStreamSynchronize(stream));

    return CUDA_SUCCESS_MOON;
}
