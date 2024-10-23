#include <bit>
#include <concepts>
#include <type_traits>

#include "../../cuda/matrix/matrix.cuh"
#include "../../cuda/mmcs/merkle_tree.cuh"
#include "../../cuda/ntt/sppark.cuh"
#include "../../cuda/opening/opening.cuh"
#include "../../cuda/quotient/quotient.cuh"
#include "../../cuda/scan/scan.cuh"
#include "../../cuda/stark/stark.cuh"
#include "../../cuda/tests/tests.cuh"
#include "../../cuda/utils/memory.cuh"
#include "../../cuda/utils/runtime.cuh"
#include "moongate_cuda.cuh"
#include "sp1tracegen.hpp"

namespace tracegen {

template<class T>
concept PopulateParams =
    requires(typename T::Val* mat, const typename T::Event& event, typename T::Cols& cols) {
        typename T::Field;
        typename T::Val;
        requires std::same_as<typename T::Val, decltype(T::Field::val)>;
        typename T::Cols;
        requires std::is_standard_layout_v<typename T::Cols>;
        typename T::Event;
        { T::write_nonce } -> std::convertible_to<bool>;
        T::event_to_row(event, cols);
        (*T::write_padding)(mat, uintptr_t {});
    };

template<PopulateParams T>
__global__ void event_to_row_kernel_alt(
    decltype(T::Field::val)* mat,
    uintptr_t width,
    uintptr_t height,
    const typename T::Event* events,
    uintptr_t nb_events
) {
    using Field = typename T::Field;
    using Cols = typename T::Cols;
    using Val = typename T::Val;
    using Event = typename T::Event;
    static const size_t COL_COUNT = sizeof(Cols) / sizeof(Val);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x) {
        Cols cols {};
        T::event_to_row(events[i], cols);
        // Write the nonce column if the flag is set.
        if constexpr (T::write_nonce) {
            cols.nonce = Field::from_canonical_u32(i).val;
        }
        // Copy populated cols to col major matrix.
        const Val* arr = std::bit_cast<Val*>(&cols);
        for (size_t j = 0; j < COL_COUNT; ++j) {
            // i, j index row, col
            // column major means the successor cell is the next row
            mat[i + j * height] = arr[j];
        }
    }
    // Process the padded section.
    for (; i < height; i += blockDim.x * gridDim.x) {
        // Write custom padding if the function pointer is set.
        if constexpr (T::write_padding != nullptr) {
            (*T::write_padding)(&mat[i], height);
        }
        // Write the nonce column if the flag is set.
        if constexpr (T::write_nonce) {
            static constexpr size_t NONCE_OFFSET =
                offsetof(Cols, nonce) / sizeof(Val);
            mat[i + NONCE_OFFSET * height] = Field::from_canonical_u32(i).val;
        }
    }
}

template<tracegen::PopulateParams T>
moongate::CudaRustError generic_populate(
    moongate::MatrixViewMutDevice<typename T::Val> mat,
    const typename T::Event* events,
    uintptr_t nb_events,
    moongate::CudaStreamHandle stream_handle
) {
    const cudaStream_t stream = std::bit_cast<cudaStream_t>(stream_handle);
    static const int M = 256;
    // Set to zero.
    CUDA_OK(cudaMemsetAsync(
        mat.values,
        0,
        mat.width * mat.height * sizeof(typename T::Val),
        stream
    ));
    tracegen::event_to_row_kernel_alt<T>
        <<<(mat.height - 1) / M + 1, M, 0, stream>>>(
            mat.values,
            mat.width,
            mat.height,
            events,
            nb_events
        );
    return CUDA_SUCCESS_MOON;
}
}  // namespace tracegen

// AddSub AIR

namespace moongate {
struct AddSubParams {
    using Field = bb31_t;
    using Val = decltype(Field::val);
    using Cols = sp1::AddSubCols<Val>;
    using Event = sp1::AluEvent;
    static constexpr auto event_to_row = sp1::add_sub::event_to_row<Field>;
    static constexpr bool write_nonce = true;
    static constexpr void (*write_padding)(Val* mat, uintptr_t height) =
        nullptr;
};

static_assert(tracegen::PopulateParams<AddSubParams>);

extern CudaRustError add_sub_populate_babybear(
    MatrixViewMutDevice<F> mat,
    const AluEvent* events,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    return tracegen::generic_populate<AddSubParams>(
        mat,
        std::bit_cast<const sp1::AluEvent*>(events),
        nb_events,
        stream_handle
    );
}
}  // namespace moongate

template __device__ void sp1::add_sub::event_to_row<bb31_t>(
    const sp1::AluEvent& event,
    sp1::AddSubCols<sp1::BabyBearMonty>& cols
);

// Bitwise AIR

namespace moongate {
struct BitwiseParams {
    using Field = bb31_t;
    using Val = decltype(Field::val);
    using Cols = sp1::BitwiseCols<Val>;
    using Event = sp1::AluEvent;
    static constexpr auto event_to_row = sp1::bitwise::event_to_row<Field>;
    static constexpr bool write_nonce = true;
    static constexpr void (*write_padding)(Val* mat, uintptr_t height) =
        nullptr;
};

static_assert(tracegen::PopulateParams<BitwiseParams>);

extern CudaRustError bitwise_populate_babybear(
    MatrixViewMutDevice<F> mat,
    const AluEvent* events,
    uintptr_t nb_events,
    CudaStreamHandle stream
) {
    return tracegen::generic_populate<BitwiseParams>(
        mat,
        std::bit_cast<const sp1::AluEvent*>(events),
        nb_events,
        stream
    );
}
}  // namespace moongate

template __device__ void sp1::bitwise::event_to_row<bb31_t>(
    const sp1::AluEvent& event,
    sp1::BitwiseCols<sp1::BabyBearMonty>& cols
);

// Lt AIR

namespace moongate {
struct LtParams {
    using Field = bb31_t;
    using Val = decltype(Field::val);
    using Cols = sp1::LtCols<Val>;
    using Event = sp1::AluEvent;
    static constexpr auto event_to_row = sp1::lt::event_to_row<Field>;
    static constexpr bool write_nonce = true;
    static constexpr void (*write_padding)(Val* mat, uintptr_t height) =
        nullptr;
};

static_assert(tracegen::PopulateParams<LtParams>);

extern CudaRustError lt_populate_babybear(
    MatrixViewMutDevice<F> mat,
    const AluEvent* events,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    return tracegen::generic_populate<LtParams>(
        mat,
        std::bit_cast<const sp1::AluEvent*>(events),
        nb_events,
        stream_handle
    );
}
}  // namespace moongate

template __device__ void sp1::lt::event_to_row<bb31_t>(
    const sp1::AluEvent& event,
    sp1::LtCols<sp1::BabyBearMonty>& cols
);

// ShiftLeft AIR

namespace moongate {
struct ShiftLeftParams {
    using Field = bb31_t;
    using Val = decltype(Field::val);
    using Cols = sp1::ShiftLeftCols<Val>;
    using Event = sp1::AluEvent;
    static constexpr auto event_to_row = sp1::sll::event_to_row<Field>;
    static constexpr bool write_nonce = true;

    __device__ static void write_padding(Val* mat, uintptr_t height) {
        mat[offsetof(Cols, shift_by_n_bits[0]) / sizeof(Val) * height] =
            Field::one().val;
        mat[offsetof(Cols, shift_by_n_bytes[0]) / sizeof(Val) * height] =
            Field::one().val;
        mat[offsetof(Cols, bit_shift_multiplier) / sizeof(Val) * height] =
            Field::one().val;
    }
};

static_assert(tracegen::PopulateParams<ShiftLeftParams>);

extern CudaRustError sll_populate_babybear(
    MatrixViewMutDevice<F> mat,
    const AluEvent* events,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    return tracegen::generic_populate<ShiftLeftParams>(
        mat,
        std::bit_cast<const sp1::AluEvent*>(events),
        nb_events,
        stream_handle
    );
}
}  // namespace moongate

template __device__ void sp1::sll::event_to_row<bb31_t>(
    const sp1::AluEvent& event,
    sp1::ShiftLeftCols<sp1::BabyBearMonty>& cols
);

// ShiftRight AIR

namespace moongate {
struct ShiftRightParams {
    using Field = bb31_t;
    using Val = decltype(Field::val);
    using Cols = sp1::ShiftRightCols<Val>;
    using Event = sp1::AluEvent;
    static constexpr auto event_to_row = sp1::sr::event_to_row<Field>;
    static constexpr bool write_nonce = true;

    __device__ static void write_padding(Val* mat, uintptr_t height) {
        mat[offsetof(Cols, shift_by_n_bits[0]) / sizeof(Val) * height] =
            Field::one().val;
        mat[offsetof(Cols, shift_by_n_bytes[0]) / sizeof(Val) * height] =
            Field::one().val;
    }
};

static_assert(tracegen::PopulateParams<ShiftRightParams>);

extern CudaRustError sr_populate_babybear(
    MatrixViewMutDevice<F> mat,
    const AluEvent* events,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    return tracegen::generic_populate<ShiftRightParams>(
        mat,
        std::bit_cast<const sp1::AluEvent*>(events),
        nb_events,
        stream_handle
    );
}
}  // namespace moongate

template __device__ void sp1::sr::event_to_row<bb31_t>(
    const sp1::AluEvent& event,
    sp1::ShiftRightCols<sp1::BabyBearMonty>& cols
);

// CPU AIR

namespace moongate {
struct CpuParams {
    using Field = bb31_t;
    using Val = decltype(Field::val);
    using Cols = sp1::CpuCols<Val>;
    using Event = sp1::CpuEventFfi;
    static constexpr auto event_to_row = sp1::cpu::event_to_row<Field>;
    static constexpr bool write_nonce = false;

    __device__ static void write_padding(Val* mat, uintptr_t height) {
        mat[offsetof(Cols, selectors.imm_b) / sizeof(Val) * height] =
            Field::one().val;
        mat[offsetof(Cols, selectors.imm_c) / sizeof(Val) * height] =
            Field::one().val;
    }
};

static_assert(tracegen::PopulateParams<CpuParams>);

extern CudaRustError cpu_populate_babybear(
    MatrixViewMutDevice<F> mat,
    const CpuEventFfi* events,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    return tracegen::generic_populate<CpuParams>(
        mat,
        std::bit_cast<const sp1::CpuEventFfi*>(events),
        nb_events,
        stream_handle
    );
}
}  // namespace moongate

template __device__ void sp1::cpu::event_to_row<bb31_t>(
    const sp1::CpuEventFfi& event,
    sp1::CpuCols<sp1::BabyBearMonty>& cols
);
