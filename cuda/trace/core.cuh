#pragma once

#include "../fields/bb31_t.cuh"
#include "../matrix/matrix.cuh"
#include "../utils/runtime.cuh"
#include "add_sub.hpp"
#include "moongate-core-sys-cbindgen.hpp"
#include "sp1-core-machine-sys-cbindgen.hpp"

using namespace moongate;

namespace moongate::add_sub {
template<class T>
__global__ void generate_trace_kernel(
    MatrixViewMutDevice<T> trace,
    const sp1_core_machine_sys::AluEvent* events,
    uintptr_t nb_events
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x) {
        sp1_core_machine_sys::AddSubCols<T> cols;
        sp1_core_machine_sys::add_sub::event_to_row<T>(events[i], cols);
    }
}

extern "C" rustCudaError_t generate_trace(
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
    generate_trace_kernel<bb31_t>
        <<<1, M, 0, stream>>>(trace, events, nb_events);

    return CUDA_SUCCESS_MOON;
}
}  // namespace moongate::add_sub
