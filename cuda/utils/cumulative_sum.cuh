#pragma once

#include <stdio.h>

#include <cuda/atomic>

#include "../fields/bb31_t.cuh"

namespace cumulative_sum {

template<class F, class EF7>
__host__ __device__ uint8_t lift_x(EF7* x, EF7* x_out, EF7* y_out) {
    EF7 x_start = *x;
#pragma unroll(1)
    for (uint32_t offset = 0; offset < 256; offset++) {
        EF7 x_trial = x_start.universal_hash();
        EF7 y_sq = x_trial.curve_formula();
        F y_sq_pow_r = y_sq.pow_r();
        F is_square = y_sq_pow_r ^ 1006632960;
        if (is_square == F::one()) {
            EF7 y = y_sq.sqrt(y_sq_pow_r);
            if (y.is_exception()) {
                x_start += F::from_canonical_u32(1 << 16);
                continue;
            }
            if (y.is_send()) {
                y = EF7::zero() - y;
            }

            *x_out = x_trial;
            *y_out = y;
            return offset;
        }
        x_start += F::from_canonical_u32(1 << 16);
    }
    assert(false);
}

extern "C" uint8_t lift_x_device(
    bb31_septic_extension_t* x,
    bb31_septic_extension_t* x_out,
    bb31_septic_extension_t* y_out
) {
    // printf("points: {}", points);
    // printf("inside\n");
    // printf("x: {}", *x);
    return lift_x<bb31_t, bb31_septic_extension_t>(x, x_out, y_out);
}

template<class F, class EF7, class P>
__global__ void populate_points(
    uint32_t* memory_image,
    size_t n,
    P* out,
    uint32_t interaction_kind
) {
#pragma unroll(1)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        EF7 x_start;
        {
            x_start.value[0] = F::from_canonical_u32(interaction_kind << 24);
            x_start.value[1] = F::zero();
            x_start.value[2] = F::from_canonical_u32(memory_image[i * 2]);
            x_start.value[3] =
                F::from_canonical_u32(memory_image[i * 2 + 1] & 255);
            x_start.value[4] =
                F::from_canonical_u32((memory_image[i * 2 + 1] >> 8) & 255);
            x_start.value[5] =
                F::from_canonical_u32((memory_image[i * 2 + 1] >> 16) & 255);
            x_start.value[6] =
                F::from_canonical_u32((memory_image[i * 2 + 1] >> 24) & 255);
        }
        EF7 x_out;
        EF7 y;
        uint8_t offset = lift_x<F, EF7>(&x_start, &x_out, &y);
        y = EF7::zero() - y;

        out[i] = P(x_out, y);
    }
}

extern "C" rustCudaError_t compute_initial_global_cumulative_sum(
    uint32_t* memory_image,
    size_t n,
    bb31_septic_curve_t* out,
    uint32_t interaction_kind,
    CudaStreamHandle stream_handle
) {
    // bb31_septic_curve_t[n] points;
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

    bb31_septic_curve_t* points;
    // printf("inside\n");
    CUDA_OK(
        cudaMallocAsync(&points, sizeof(bb31_septic_curve_t) * (n + 1), stream)
    );
    // printf("allocated points\n");
    bb31_septic_curve_t start_point = bb31_septic_curve_t::start_point();
    CUDA_OK(cudaMemcpyAsync(
        &points[n],
        &start_point,
        sizeof(bb31_septic_curve_t),
        cudaMemcpyHostToDevice,
        stream
    ));
    // printf("set start point\n");
    static const int M = 128;
    populate_points<bb31_t, bb31_septic_extension_t, bb31_septic_curve_t>
        <<<(n - 1) / M + 1, M, 0, stream>>>(
            memory_image,
            n,
            points,
            interaction_kind
        );
    // printf("populated\n");
    CUDA_OK(cudaStreamSynchronize(stream));
    // printf("synced\n");

    // Call scan to sum all the points.
    ScanTemplateLarge(out, points, n + 1, stream);
    // printf("scanned\n");
    CUDA_OK(cudaStreamSynchronize(stream));
    // printf("synced\n");

    CUDA_OK(cudaFreeAsync(points, stream));

    // printf("done initial global cumulative sum\n");

    return CUDA_SUCCESS_MOON;
}

}  // namespace cumulative_sum
