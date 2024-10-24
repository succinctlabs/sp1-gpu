#pragma once

#include <bit>

#include "../fields/bb31_extension_t.cuh"
#include "../fields/bb31_t.cuh"
#include "interaction.cuh"
#include "moongate_cuda_cbindgen.hpp"
#include "permutation.cu"

namespace moongate {

void populate_permutation_rows_flattened(
    DeviceInteractionsView<BabyBear> interactions,
    MatrixViewMutDevice<BabyBear> permutation,
    MatrixViewDevice<BabyBear> preprocessed,
    MatrixViewDevice<BabyBear> main,
    BinomialExtensionField<BabyBear, 4> global_alpha,
    BinomialExtensionField<BabyBear, 4> global_beta,
    BinomialExtensionField<BabyBear, 4> local_alpha,
    BinomialExtensionField<BabyBear, 4> local_beta,
    uintptr_t batch_size,
    uintptr_t num_blocks,
    uintptr_t num_threads_per_block,
    CudaStreamHandle stream
) {
    assert(!permutation.row_major);
    assert(!main.row_major);
    assert(!preprocessed.row_major);
    assert(permutation.height == main.height);
    PopulatePermutationRowsFlattened<<<
        num_blocks,
        num_threads_per_block,
        0,
        std::bit_cast<cudaStream_t>(stream)>>>(
        std::bit_cast<Interactions<bb31_t>>(interactions),
        std::bit_cast<Matrix<bb31_t>>(permutation),
        std::bit_cast<Matrix<bb31_t>>(preprocessed),
        std::bit_cast<Matrix<bb31_t>>(main),
        std::bit_cast<bb31_extension_t>(global_alpha),
        std::bit_cast<bb31_extension_t>(global_beta),
        std::bit_cast<bb31_extension_t>(local_alpha),
        std::bit_cast<bb31_extension_t>(local_beta),
        batch_size
    );
}

}  // namespace moongate