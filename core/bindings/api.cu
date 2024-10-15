#include "../../cuda/mmcs/merkle_tree.cuh"
#include "../../cuda/matrix/matrix.cuh"
#include "../../cuda/utils/memory.cuh"
#include "../../cuda/ntt/sppark.cuh"
#include "../../cuda/utils/runtime.cuh"
#include "../../cuda/tests/tests.cuh"
#include "../../cuda/stark/stark.cuh"
#include "../../cuda/scan/scan.cuh"
#include "../../cuda/quotient/quotient.cuh"
#include "../../cuda/opening/opening.cuh"

#include "moongate_cuda.cuh"
#include "sp1tracegen.hpp"

namespace moongate {

static const size_t ADD_SUB_COL_CT = sizeof(sp1::AddSubCols<sp1::BabyBearMonty>) / sizeof(sp1::BabyBearMonty);

__global__ void add_sub_events_to_rows_babybear_kernel(MatrixViewMutDevice<F> mat, const AluEvent* events, uintptr_t nb_events) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nb_events) {
        sp1::AddSubCols<decltype(bb31_t::val)> cols{};
        // if mat is a row major matrix:
        // sp1::AddSubCols<decltype(bb31_t::val)>* cols = reinterpret_cast<sp1::AddSubCols<decltype(bb31_t::val)>*>(&mat.values[i * ADD_SUB_COL_CT]);
        sp1::add_sub::event_to_row<bb31_t>(*reinterpret_cast<const sp1::AluEvent*>(&events[i]), cols);
        // Copy populated cols to col major matrix.
        const F* arr = reinterpret_cast<F*>(&cols);
        for (size_t j = 0; j < ADD_SUB_COL_CT; ++j) {
            // i, j index row, col
            // column major means the successor cell is the next row
            mat.values[i + j * mat.height] = arr[j];
        }
    }
}

extern "C" CudaRustError add_sub_events_to_rows_babybear(
    MatrixViewMutDevice<F> mat,
    const AluEvent* events,
    uintptr_t nb_events
) {
    static const int M = 256;
    // Set to zero.
    cudaError_t code = cudaMemset(mat.values, 0, mat.width * mat.height * sizeof(F));
    if (code != cudaSuccess) {
        return CudaRustError{message : cudaGetErrorString(code)};
    }
    add_sub_events_to_rows_babybear_kernel<<<(nb_events - 1)/M + 1, M>>>(mat, events, nb_events);
    return CudaRustError{message: cudaGetErrorString(cudaSuccess)};
}

}  // namespace moongate