
#pragma once

#include <cuda/atomic>
#include <ntt/ntt.cuh>

#include "../fields/bb31_extension_t.cuh"
#include "../fields/bb31_t.cuh"

namespace fri_batch {

template<typename F>
__device__ __forceinline__ void atomicAdd(F* dst, F val) {
    cuda::atomic_ref<F, cuda::thread_scope_device> atomic_ref(*dst);
    F old_val = atomic_ref.load(cuda::memory_order_relaxed);
    F desired;

    do {
        desired = old_val + val;
    } while (!atomic_ref.compare_exchange_weak(
        old_val,
        desired,
        cuda::memory_order_relaxed
    ));
}

template<typename F, typename EF>
__global__ void batchFriKernel(
    EF* reducedOpenings,
    const F* polynomialBatch,
    const EF* evaluations,
    const F domainGenerator,
    const F shift,
    const EF evaluationPoint,
    const EF batchingChallenge,
    const EF batchingChallengeOffset,
    size_t width,
    size_t logHeight
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t height = 1U << logHeight;
    if (i >= height)
        return;

    F domainPoint = domainGenerator
        ^ (bit_rev(blockIdx.x * blockDim.x + threadIdx.x, logHeight));
    domainPoint *= shift;
    EF batchingPower = batchingChallengeOffset;
    EF inverseDenom = (evaluationPoint - domainPoint).reciprocal();
    EF accumulator = EF::zero();
    for (size_t j = 0; j < width; j++) {
        // Compute batch_value = alpha^i ((p(z) - p(x) / (z - x)).
        EF batchValue = evaluations[j] - polynomialBatch[j * height + i];
        batchValue *= batchingPower;
        accumulator += batchValue;
        batchingPower *= batchingChallenge;
    }
    accumulator *= inverseDenom;
    // Add the results to the reduced openings.
    atomicAdd(&reducedOpenings[i], accumulator);
    // reducedOpenings[i] += accumulator;
}

extern "C" void batchFri(
    bb31_extension_t* reducedOpenings,
    const bb31_t* polynomialBatch,
    const bb31_extension_t* evaluations,
    const bb31_t domainGenerator,
    const bb31_t shift,
    const bb31_extension_t evaluationPoint,
    const bb31_extension_t batchingChallenge,
    const bb31_extension_t batchingChallengeOffset,
    size_t width,
    size_t logHeight,
    cudaStream_t stream
) {
    size_t blockDim = 512;
    size_t height = 1U << logHeight;
    size_t gridDim = (height - 1) / blockDim + 1;

    batchFriKernel<<<gridDim, blockDim, 0, stream>>>(
        reducedOpenings,
        polynomialBatch,
        evaluations,
        domainGenerator,
        shift,
        evaluationPoint,
        batchingChallenge,
        batchingChallengeOffset,
        width,
        logHeight
    );
}
};  // namespace fri_batch