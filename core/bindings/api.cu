#include <bit>

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

namespace tracegen_kernels {

template<class Field, template<class> class Cols, class Event>
using EventToRow = void (*)(const Event &event, Cols<decltype(Field::val)> &cols);

template<
    class Field,
    template<class>
    class Cols,
    class Event,
    EventToRow<Field, Cols, Event> event_to_row,
    bool write_nonce = false>
__global__ void event_to_row_kernel(
    decltype(Field::val)* mat,
    uintptr_t width,
    uintptr_t height,
    const Event* events,
    uintptr_t nb_events
) {
    using Val = decltype(Field::val);
    const size_t COL_COUNT = sizeof(Cols<Val>) / sizeof(Val);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < nb_events; i += blockDim.x * gridDim.x) {
        Cols<Val> cols {};
        event_to_row(events[i], cols);
        if (write_nonce) {
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
    if (write_nonce) {
        for (; i < height; i += blockDim.x * gridDim.x) {
            static const size_t NONCE_OFFSET =
                offsetof(Cols<Val>, nonce) / sizeof(Val);
            mat[i + NONCE_OFFSET * height] = Field::from_canonical_u32(i).val;
        }
    }
}
}  // namespace tracegen_kernels

namespace moongate {
// Not an implementation of an extern function.
template<
    class Field,
    template<class>
    class Cols,
    class Event,
    tracegen_kernels::EventToRow<Field, Cols, Event> event_to_row,
    bool write_nonce = false>
CudaRustError generic_populate_babybear(
    MatrixViewMutDevice<F> mat,
    const Event* events,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    const cudaStream_t stream = std::bit_cast<cudaStream_t>(stream_handle);
    static const int M = 256;
    // Set to zero.
    cudaError_t code = cudaMemsetAsync(
        mat.values,
        0,
        mat.width * mat.height * sizeof(F),
        stream
    );
    if (code != cudaSuccess) {
        return CudaRustError {message: cudaGetErrorString(code)};
    }
    tracegen_kernels::event_to_row_kernel<Field, Cols, Event, event_to_row, write_nonce>
        <<<(mat.height - 1) / M + 1, M, 0, stream>>>(
            mat.values,
            mat.width,
            mat.height,
            events,
            nb_events
        );
    return CudaRustError {message: cudaGetErrorString(cudaSuccess)};
}
}  // namespace moongate

namespace moongate {
extern CudaRustError add_sub_populate_babybear(
    MatrixViewMutDevice<F> mat,
    const AluEvent* events,
    uintptr_t nb_events,
    CudaStreamHandle stream
) {
    return generic_populate_babybear<
        bb31_t,
        sp1::AddSubCols,
        sp1::AluEvent,
        sp1::add_sub::event_to_row<bb31_t>,
        true>(
        mat,
        std::bit_cast<const sp1::AluEvent*>(events),
        nb_events,
        stream
    );
}
}  // namespace moongate

template __device__ void sp1::add_sub::event_to_row<bb31_t>(
    const sp1::AluEvent& event,
    sp1::AddSubCols<sp1::BabyBearMonty>& cols
);

namespace moongate {
extern CudaRustError bitwise_populate_babybear(
    MatrixViewMutDevice<F> mat,
    const AluEvent* events,
    uintptr_t nb_events,
    CudaStreamHandle stream
) {
    return generic_populate_babybear<
        bb31_t,
        sp1::BitwiseCols,
        sp1::AluEvent,
        sp1::bitwise::event_to_row<bb31_t>,
        true>(
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

namespace moongate {
extern CudaRustError lt_populate_babybear(
    MatrixViewMutDevice<F> mat,
    const AluEvent* events,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    return generic_populate_babybear<
        bb31_t,
        sp1::LtCols,
        sp1::AluEvent,
        sp1::lt::event_to_row<bb31_t>,
        true>(
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

namespace moongate {
extern CudaRustError sll_populate_babybear(
    MatrixViewMutDevice<F> mat,
    const AluEvent* events,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    return generic_populate_babybear<
        bb31_t,
        sp1::ShiftLeftCols,
        sp1::AluEvent,
        sp1::sll::event_to_row<bb31_t>,
        true>(
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

namespace moongate {
extern CudaRustError sr_populate_babybear(
    MatrixViewMutDevice<F> mat,
    const AluEvent* events,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    return generic_populate_babybear<
        bb31_t,
        sp1::ShiftRightCols,
        sp1::AluEvent,
        sp1::sr::event_to_row<bb31_t>,
        true>(
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

namespace moongate {
extern CudaRustError cpu_populate_babybear(
    MatrixViewMutDevice<F> mat,
    const CpuEventFfi* events,
    uintptr_t nb_events,
    CudaStreamHandle stream_handle
) {
    return generic_populate_babybear<
        bb31_t,
        sp1::CpuCols,
        sp1::CpuEventFfi,
        sp1::cpu::event_to_row<bb31_t>>(
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
