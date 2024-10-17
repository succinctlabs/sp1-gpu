#pragma once

#include <bit>

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
        std::bit_cast<Interactions<bb31_t> const>(interactions),
        std::bit_cast<Matrix<bb31_t>>(permutation),
        std::bit_cast<Matrix<bb31_t> const>(preprocessed),
        std::bit_cast<Matrix<bb31_t> const>(main),
        std::bit_cast<bb31_extension_t const>(global_alpha),
        std::bit_cast<bb31_extension_t const>(global_beta),
        std::bit_cast<bb31_extension_t const>(local_alpha),
        std::bit_cast<bb31_extension_t const>(local_beta),
        batch_size
    );
}

}  // namespace moongate