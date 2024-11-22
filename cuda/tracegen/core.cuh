#pragma once

#include "../fields/bb31_t.cuh"
#include "../fields/bb31_curve_t.cuh"
#include "../matrix/matrix.cuh"
#include "../utils/runtime.cuh"
#include "../matrix/transpose.cuh"
#include "add_sub.hpp"
#include "memory_local.hpp"
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

        const T* arr = std::bit_cast<T*>(&cols);
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
    const cudaStream_t stream = std::bit_cast<cudaStream_t>(stream_handle);
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

template<class F, class EF7>
__global__ void core_memory_local_generate_trace_decompress_kernel(
    MatrixViewMutDevice<F> trace,
    // uintptr_t height,
    // sp1_core_machine_sys::MemoryLocalCols<F> *cols,
    const sp1_core_machine_sys::MemoryLocalEvent* events,
    uintptr_t nb_events
) {
    static const size_t MEMORY_LOCAL_COLUMNS =
        sizeof(sp1_core_machine_sys::MemoryLocalCols<F>) / sizeof(F);
    
    static const size_t SINGLE_MEMORY_LOCAL_COLUMNS =
        sizeof(sp1_core_machine_sys::SingleMemoryLocal<F>) / sizeof(F);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll(1)
    for (; i < trace.height ; i += blockDim.x * gridDim.x) {
        // ok so we're on the ith row
        bb31_septic_curve_t sum = bb31_septic_curve_t();
        if (i == 0) {
            sum = bb31_septic_curve_t::start_point();
        }
        sp1_core_machine_sys::MemoryLocalCols<F> cols;
        F *cols_arr = std::bit_cast<F*>(&cols);
        for (int k = 0; k < MEMORY_LOCAL_COLUMNS; k++) {
            cols_arr[k] = F::zero();
        }
        for(int j = 0 ; j < 4 ; j++) {
            int event_idx = 4 * i + j;
            if (event_idx < nb_events) {
                sp1_core_machine_sys::memory_local::event_to_row<F, EF7>(&events[event_idx], &cols.memory_local_entries[j]);
                {
                    cols.global_accumulation_cols.cumulative_sum[2 * j][0] = cols.memory_local_entries[j].initial_global_interaction_cols.x_coordinate;
                    cols.global_accumulation_cols.cumulative_sum[2 * j][1] = cols.memory_local_entries[j].initial_global_interaction_cols.y_coordinate;
                    bb31_septic_curve_t point = bb31_septic_curve_t(cols.memory_local_entries[j].initial_global_interaction_cols.x_coordinate._0, cols.memory_local_entries[j].initial_global_interaction_cols.y_coordinate._0);
                    sum += point;
                }
                {
                    cols.global_accumulation_cols.cumulative_sum[2 * j + 1][0] = cols.memory_local_entries[j].final_global_interaction_cols.x_coordinate;
                    cols.global_accumulation_cols.cumulative_sum[2 * j + 1][1] = cols.memory_local_entries[j].final_global_interaction_cols.y_coordinate;
                    bb31_septic_curve_t point = bb31_septic_curve_t(cols.memory_local_entries[j].final_global_interaction_cols.x_coordinate._0, cols.memory_local_entries[j].final_global_interaction_cols.y_coordinate._0);
                    sum += point;
                }
            }
        }
        for (int k = 0 ; k < 7 ; k++) {
            cols.global_accumulation_cols.initial_digest[0]._0[k] = sum.x.value[k];
            cols.global_accumulation_cols.initial_digest[1]._0[k] = sum.y.value[k];
        }
        const F* arr = std::bit_cast<F*>(&cols);
        for (size_t k = 0; k < MEMORY_LOCAL_COLUMNS ; ++k) {
            trace.values[i + k * trace.height] = arr[k];
        }
    }

   /**int i = blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll(1)
    for (; i < height ; i += blockDim.x * gridDim.x) {
        // ok so we're on the ith row
        // sp1_core_machine_sys::MemoryLocalCols<F> *cols = std::bit_cast<sp1_core_machine_sys::MemoryLocalCols<F>*>(trace.values + i * MEMORY_LOCAL_COLUMNS);
        bb31_septic_curve_t sum = bb31_septic_curve_t();
        if (i == 0) {
            sum = bb31_septic_curve_t::start_point();
        }
        for(int j = 0 ; j < 4 ; j++) {
            int event_idx = 4 * i + j;
            if (event_idx < nb_events) {
                sp1_core_machine_sys::memory_local::event_to_row<F, EF7>(&events[event_idx], &cols[i].memory_local_entries[j]);
                {
                    cols[i].global_accumulation_cols.cumulative_sum[2 * j][0] = cols[i].memory_local_entries[j].initial_global_interaction_cols.x_coordinate;
                    cols[i].global_accumulation_cols.cumulative_sum[2 * j][1] = cols[i].memory_local_entries[j].initial_global_interaction_cols.y_coordinate;
                    bb31_septic_curve_t point = bb31_septic_curve_t(cols[i].memory_local_entries[j].initial_global_interaction_cols.x_coordinate._0, cols[i].memory_local_entries[j].initial_global_interaction_cols.y_coordinate._0);
                    sum += point;
                }
                {
                    cols[i].global_accumulation_cols.cumulative_sum[2 * j + 1][0] = cols[i].memory_local_entries[j].final_global_interaction_cols.x_coordinate;
                    cols[i].global_accumulation_cols.cumulative_sum[2 * j + 1][1] = cols[i].memory_local_entries[j].final_global_interaction_cols.y_coordinate;
                    bb31_septic_curve_t point = bb31_septic_curve_t(cols[i].memory_local_entries[j].final_global_interaction_cols.x_coordinate._0, cols[i].memory_local_entries[j].final_global_interaction_cols.y_coordinate._0);
                    sum += point;
                }
            }
        }
        for (int k = 0 ; k < 7 ; k++) {
            cols[i].global_accumulation_cols.initial_digest[0]._0[k] = sum.x.value[k];
            cols[i].global_accumulation_cols.initial_digest[1]._0[k] = sum.y.value[k];
        }
    }
    */
}

template<class F, class EF7>
__global__ void core_memory_local_generate_trace_finalize_kernel(
    MatrixViewMutDevice<F> trace,
    bb31_septic_curve_t *cumulative_sums,
    uintptr_t nb_events
) {
    static const size_t MEMORY_LOCAL_COLUMNS =
        sizeof(sp1_core_machine_sys::MemoryLocalCols<F>) / sizeof(F);
    
    static const size_t SINGLE_MEMORY_LOCAL_COLUMNS =
        sizeof(sp1_core_machine_sys::SingleMemoryLocal<F>) / sizeof(F);

    /*int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < trace.height ; i += blockDim.x * gridDim.x) {
        // ok so we're on the ith row
        for(int j = 0 ; j < 4 ; j++) {
            int event_idx = 4 * i + j;
            if (event_idx < nb_events) {
                sp1_core_machine_sys::SingleMemoryLocal<F> cols;
                sp1_core_machine_sys::memory_local::event_to_row<F, EF7>(events[event_idx], cols);
                const F* arr = std::bit_cast<F*>(&cols);
                for (size_t k = 0; k < SINGLE_MEMORY_LOCAL_COLUMNS ; ++k) {
                    trace.values[i + (k + j * SINGLE_MEMORY_LOCAL_COLUMNS) * trace.height] = arr[k];
                }
            }
        }
    }*/

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    #pragma unroll(1)
    for (; i < trace.height ; i += blockDim.x * gridDim.x) {
        sp1_core_machine_sys::MemoryLocalCols<F> cols;
        F* temp_arr = reinterpret_cast<F*>(&cols);
        for(int j = 0 ; j < MEMORY_LOCAL_COLUMNS ; j++) {
            temp_arr[j] = trace.values[i + j * trace.height];
        }
        
        bb31_septic_curve_t sum = cumulative_sums[i];
        
        for(int j = 3 ; j >= 0 ; j--) {
            int event_idx = 4 * i + j;
            {
                {
                    bb31_septic_extension_t point_x = bb31_septic_extension_t(cols.memory_local_entries[j].final_global_interaction_cols.x_coordinate._0);
                    bb31_septic_extension_t point_y = bb31_septic_extension_t(cols.memory_local_entries[j].final_global_interaction_cols.y_coordinate._0) * (bb31_t::zero() - bb31_t::one());
                    bb31_septic_curve_t point = bb31_septic_curve_t(point_x, point_y);
                    for (int k = 0 ; k < 7 ; k++) {
                        cols.global_accumulation_cols.cumulative_sum[2 * j + 1][0]._0[k] = sum.x.value[k];
                        cols.global_accumulation_cols.cumulative_sum[2 * j + 1][1]._0[k] = sum.y.value[k];
                    }
                    sum += point;
                    if (event_idx >= nb_events) {
                        bb31_septic_curve_t dummy = bb31_septic_curve_t::dummy_point();
                        for (int k = 0 ; k < 7 ; k++) {
                            cols.memory_local_entries[j].final_global_interaction_cols.x_coordinate._0[k] = dummy.x.value[k];
                            cols.memory_local_entries[j].final_global_interaction_cols.y_coordinate._0[k] = dummy.y.value[k];
                        }
                    }
                }

                {
                    bb31_septic_extension_t point_x = bb31_septic_extension_t(cols.memory_local_entries[j].initial_global_interaction_cols.x_coordinate._0);
                    bb31_septic_extension_t point_y = bb31_septic_extension_t(cols.memory_local_entries[j].initial_global_interaction_cols.y_coordinate._0) * (bb31_t::zero() - bb31_t::one());
                    bb31_septic_curve_t point = bb31_septic_curve_t(point_x, point_y);
                    for (int k = 0 ; k < 7 ; k++) {
                        cols.global_accumulation_cols.cumulative_sum[2 * j][0]._0[k] = sum.x.value[k];
                        cols.global_accumulation_cols.cumulative_sum[2 * j][1]._0[k] = sum.y.value[k];
                    }
                    sum += point;
                    if (event_idx >= nb_events) {
                        bb31_septic_curve_t dummy = bb31_septic_curve_t::dummy_point();
                        for (int k = 0 ; k < 7 ; k++) {
                            cols.memory_local_entries[j].initial_global_interaction_cols.x_coordinate._0[k] = dummy.x.value[k];
                            cols.memory_local_entries[j].initial_global_interaction_cols.y_coordinate._0[k] = dummy.y.value[k];
                        }
                    }
                }
            }
        }
        for (int k = 0 ; k < 7 ; k++) {
            cols.global_accumulation_cols.initial_digest[0]._0[k] = sum.x.value[k];
            cols.global_accumulation_cols.initial_digest[1]._0[k] = sum.y.value[k];
        }
        for(int j = 0 ; j < 8 ; j++) {
            if(4 * i + j / 2 < nb_events) { 
                for(int k = 0 ; k < 7 ; k++) {
                    cols.global_accumulation_cols.sum_checker[j]._0[k] = bb31_t::zero();
                }
            }
            else {
                bb31_septic_curve_t dummy = bb31_septic_curve_t::dummy_point();
                bb31_septic_curve_t digest = bb31_septic_curve_t(cols.global_accumulation_cols.cumulative_sum[j][0]._0, cols.global_accumulation_cols.cumulative_sum[j][1]._0);
                bb31_septic_extension_t sum_checker_x = bb31_septic_curve_t::sum_checker_x(digest, dummy, digest);
                for(int k = 0 ; k < 7 ; k++) {
                    cols.global_accumulation_cols.sum_checker[j]._0[k] = sum_checker_x.value[k];
                }
            }
        }
        
        F* final_temp = reinterpret_cast<F*>(&cols);
        for(int j = 0 ; j < MEMORY_LOCAL_COLUMNS ; j++) {
            trace.values[i + j * trace.height] = final_temp[j];
        }
    }
}

extern "C" rustCudaError_t core_memory_local_generate_trace_round_3(
    MatrixViewMutDevice<bb31_t> trace, // this is still column major!
    bb31_septic_curve_t *cumulative_sums,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    const cudaStream_t stream = std::bit_cast<cudaStream_t>(stream_handle);

    static const int M = 16;
   
    uintptr_t real_rows = (nb_events + 3) / 4;
    if(nb_events == 0) {
        real_rows = 1;
    }

    size_t stackSize;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    // printf("Current stack size: %zu bytes\n", stackSize);

    cudaDeviceSetLimit(cudaLimitStackSize, 4096);

    // sp1_core_machine_sys::MemoryLocalCols<bb31_t> *cols = std::bit_cast<sp1_core_machine_sys::MemoryLocalCols<bb31_t>*>(trace.values);
    core_memory_local_generate_trace_finalize_kernel<bb31_t, bb31_septic_extension_t><<<(trace.height - 1) / M + 1, M, 0, stream>>>(
        // trace,
        // trace.height,
        trace,
        cumulative_sums,
        nb_events
    );

    return CUDA_SUCCESS_MOON;

}

extern "C" rustCudaError_t core_memory_local_generate_trace_round_2(
    MatrixViewMutDevice<bb31_t> trace,
    bb31_septic_curve_t *cumulative_sums,
    CudaStreamHandle stream_handle
) {
    // printf("trace.height: %zu\n", trace.height);
    // printf("trace.width: %zu\n", trace.width);
    // printf("trace.is_row_major: %d\n", trace.row_major);

    // Get the stream.
    const cudaStream_t stream = std::bit_cast<cudaStream_t>(stream_handle);

    // Select the right set of columns from the trace.
    Matrix<bb31_t> initial_digest_trace_col_major;
    initial_digest_trace_col_major.values = trace.values + trace.height * sp1_core_machine_sys::MEMORY_LOCAL_INITIAL_DIGEST_POS_COPY;
    initial_digest_trace_col_major.width = 14;
    initial_digest_trace_col_major.height = trace.height;
    initial_digest_trace_col_major.row_major = false;


    // Copy initial digest trace back to host for debugging
    bb31_t* host_initial_digest_col_major = new bb31_t[initial_digest_trace_col_major.width * initial_digest_trace_col_major.height];
    CUDA_OK(cudaMemcpyAsync(
        host_initial_digest_col_major,
        initial_digest_trace_col_major.values,
        initial_digest_trace_col_major.width * initial_digest_trace_col_major.height * sizeof(bb31_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_OK(cudaStreamSynchronize(stream));

    // printf("First value in column-major initial digest: %d\n", host_initial_digest_col_major[0].as_canonical_u32());
    // printf("Second value in column-major initial digest: %d\n", host_initial_digest_col_major[1].as_canonical_u32());
    
    delete[] host_initial_digest_col_major;

    // Allocate memory for the row-major version of the initial digest trace.
    bb31_t *initial_digest_trace_row_major;
    CUDA_OK(cudaMalloc(&initial_digest_trace_row_major, sizeof(bb31_t) * 14 * trace.height));

    // Transpose the initial digest trace from column-major to row-major.
    matrix_transpose::transpose_naive(initial_digest_trace_row_major, initial_digest_trace_col_major, stream);

    // Copy row-major initial digest trace back to host for debugging
    bb31_t* host_initial_digest_row_major = new bb31_t[14 * trace.height];
    CUDA_OK(cudaMemcpyAsync(
        host_initial_digest_row_major,
        initial_digest_trace_row_major,
        14 * trace.height * sizeof(bb31_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_OK(cudaStreamSynchronize(stream));

    // printf("First value in row-major initial digest: %d\n", host_initial_digest_row_major[0].as_canonical_u32());
    // printf("Second value in row-major initial digest: %d\n", host_initial_digest_row_major[1].as_canonical_u32());
    
    delete[] host_initial_digest_row_major;

    // Cast the row-major initial digest trace to a curve.
    bb31_septic_curve_t* initial_digest_trace_row_major_curve = reinterpret_cast<bb31_septic_curve_t*>(initial_digest_trace_row_major);

    // Copy initial digest trace back to host for debugging
    bb31_septic_curve_t* host_initial_digest = new bb31_septic_curve_t[trace.height];
    CUDA_OK(cudaMemcpyAsync(
        host_initial_digest,
        initial_digest_trace_row_major_curve,
        trace.height * sizeof(bb31_septic_curve_t),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_OK(cudaStreamSynchronize(stream));
    /*printf("First initial digest: x=(%d,%d...) y=(%d,%d...)\n",
        host_initial_digest[0].x.value[0].as_canonical_u32(),
        host_initial_digest[0].x.value[1].as_canonical_u32(),
        host_initial_digest[0].y.value[0].as_canonical_u32(),
        host_initial_digest[0].y.value[1].as_canonical_u32()
    );*/


    
    // Print first few values
    /*printf("First few initial digest values:\n");
    for (int i = 0; i < std::min(5UL, trace.height); i++) {
        printf("Row %d: x=(%d,...) y=(%d,...)\n", 
            i,
            host_initial_digest[i].x.value[0].as_canonical_u32(),
            host_initial_digest[i].y.value[0].as_canonical_u32()
        );
    }*/
    
    delete[] host_initial_digest;

    // Compute the cumulative sums of the initial digest trace.
    auto error = ScanTemplate(cumulative_sums, initial_digest_trace_row_major_curve, trace.height, stream);
    // printf("Error: %s\n", error.message);

    // Free the allocated memory for the row-major initial digest trace.
    // CUDA_OK(cudaFree(initial_digest_trace_row_major));

    return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t core_memory_local_generate_trace_round_1(
    MatrixViewMutDevice<bb31_t> trace,
    const sp1_core_machine_sys::MemoryLocalEvent* events,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    const cudaStream_t stream = std::bit_cast<cudaStream_t>(stream_handle);
    CUDA_OK(cudaMemsetAsync(
        trace.values,
        0,
        trace.width * trace.height * sizeof(bb31_t),
        stream
    ));

    static const int M = 16;
   
    uintptr_t real_rows = (nb_events + 3) / 4;
    if(nb_events == 0) {
        real_rows = 1;
    }

    size_t stackSize;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    // printf("Current stack size: %zu bytes\n", stackSize);

    cudaDeviceSetLimit(cudaLimitStackSize, 4096);

    // sp1_core_machine_sys::MemoryLocalCols<bb31_t> *cols = std::bit_cast<sp1_core_machine_sys::MemoryLocalCols<bb31_t>*>(trace.values);
    core_memory_local_generate_trace_decompress_kernel<bb31_t, bb31_septic_extension_t><<<(trace.height - 1) / M + 1, M, 0, stream>>>(
        // trace,
        // trace.height,
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

        const T* arr = std::bit_cast<T*>(&cols);
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
    const cudaStream_t stream = std::bit_cast<cudaStream_t>(stream_handle);
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
