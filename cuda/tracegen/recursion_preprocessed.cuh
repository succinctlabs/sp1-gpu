#pragma once

#include "../fields/bb31_t.cuh"
#include "../matrix/matrix.cuh"
#include "../utils/runtime.cuh"
#include "alu_base.hpp"
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
        sizeof(sp1_recursion_core_sys::BaseAluInstrCols<T>) / sizeof(T);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_instructions; i += blockDim.x * gridDim.x) {
        sp1_recursion_core_sys::BaseAluInstrCols<T> cols;
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