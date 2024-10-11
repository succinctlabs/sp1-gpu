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

#include <cuda/std/array>

namespace moongate {

static const size_t ADD_SUB_ROW_LEN = sizeof(sp1::AddSubCols<sp1::BabyBearMonty>) / sizeof(sp1::BabyBearMonty);

// Compiles to a no-op with -O3 and the like.
__device__ inline cuda::std::array<uint8_t, 4> u32_to_le_bytes(uint32_t n)
{
    return {
        (uint8_t)(n >> 8 * 0),
        (uint8_t)(n >> 8 * 1),
        (uint8_t)(n >> 8 * 2),
        (uint8_t)(n >> 8 * 3),
    };
}

template <class F>
__device__ inline void populate_word_from_u32(sp1::Word<decltype(F::val)> &word, const uint32_t value)
{
    // Coercion to `uint8_t` truncates the number.
    word._0[0] = F::from_canonical_u8(value).val;
    word._0[1] = F::from_canonical_u8(value >> 8).val;
    word._0[2] = F::from_canonical_u8(value >> 16).val;
    word._0[3] = F::from_canonical_u8(value >> 24).val;
}

template <class F>
__device__ inline uint32_t populate(sp1::AddOperation<decltype(F::val)> &op, const uint32_t a_u32, const uint32_t b_u32)
{
    cuda::std::array<uint8_t, 4> a = u32_to_le_bytes(a_u32);
    cuda::std::array<uint8_t, 4> b = u32_to_le_bytes(b_u32);
    bool carry = a[0] + b[0] > 0xFF;
    op.carry[0] = F::from_bool(carry).val;
    carry = a[1] + b[1] + carry > 0xFF;
    op.carry[1] = F::from_bool(carry).val;
    carry = a[2] + b[2] + carry > 0xFF;
    op.carry[2] = F::from_bool(carry).val;

    // No range check or byte lookup yet.

    uint32_t expected = a_u32 + b_u32;
    populate_word_from_u32<F>(op.value, expected);
    return expected;
}

template <class F>
__device__ void event_to_row(const sp1::AluEvent &event, sp1::AddSubCols<decltype(F::val)> &cols)
{
    bool is_add = event.opcode == sp1::Opcode::ADD;
    cols.shard = F::from_canonical_u32(event.shard).val;
    cols.is_add = F::from_bool(is_add).val;
    cols.is_sub = F::from_bool(!is_add).val;

    auto operand_1 = is_add ? event.b : event.a;
    auto operand_2 = event.c;

    populate<F>(cols.add_operation, operand_1, operand_2);
    populate_word_from_u32<F>(cols.operand_1, operand_1);
    populate_word_from_u32<F>(cols.operand_2, operand_2);
}

__global__ void add_sub_events_to_rows_babybear_kernel(MatrixViewMutDevice<F> mat, const AluEvent* events, uintptr_t nb_events) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nb_events) {
        sp1::AddSubCols<decltype(bb31_t::val)>* cols = reinterpret_cast<sp1::AddSubCols<decltype(bb31_t::val)>*>(&mat.values[i * ADD_SUB_ROW_LEN]);
        event_to_row<bb31_t>(*reinterpret_cast<const sp1::AluEvent*>(&events[i]), *cols);
    }
}

extern "C" CudaRustError add_sub_events_to_rows_babybear(
    MatrixViewMutDevice<F> mat,
    const AluEvent* events,
    uintptr_t nb_events
) {
    const int M = 256;
    // Set to zero.
    cudaError_t code = cudaMemset(mat.values, 0, mat.width * mat.height * sizeof(F));
    if (code != cudaSuccess) {
        return CudaRustError{message : cudaGetErrorString(code)};
    }
    add_sub_events_to_rows_babybear_kernel<<<(nb_events - 1)/M + 1, M>>>(mat, events, nb_events);
    return CudaRustError{message: cudaGetErrorString(cudaSuccess)};
}

}  // namespace moongate