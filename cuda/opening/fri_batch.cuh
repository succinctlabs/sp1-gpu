
#pragma once

#include <ntt/ntt.cuh>
#include <cuda/atomic>

#include "../fields/bb31_extension_t.cuh"



namespace fri_batch {
template <typename T>
__device__ void atomicAdd(cuda::atomic_ref<T, cuda::thread_scope_device>& atomic_ref, const T&& value) {
    T old_val = atomic_ref.load(cuda::std::memory_order_relaxed);
    T new_val;

    do {
        new_val = old_val + value;
    } while (!atomic_ref.compare_exchange_weak(old_val, new_val, cuda::std::memory_order_relaxed));
}

template<typename F, typename EF> __global__ void batchFri(
    Matrix<F> leafMatrix,
    const F* polynomialBatch,
    const EF* evaluations,
    const F domainGenerator,
    const F shift,
    const EF evaluationPoint,
    const EF batchingChallenge,
    const EF batchingChallengeOffset,
    const EF batchingChallengeStride,
    size_t width,
    size_t height
) {
    // Stride loops to accumulate elements from all rows and columns.
    F domainPoint = domainGenerator^(blockIdx.x * blockDim.x + threadIdx.x);
    domainPoint *= shift;
    F domainPowerStride = domainGenerator^(blockDim.x * gridDim.x); 
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < height; i += blockDim.x * gridDim.x) {
        EF batchingPower = batchingChallengeOffset;
        EF inverseDenom = EF::one() / (EF(domainPoint) - evaluationPoint);
        EF accumulator = EF::zero(); 
        for (size_t j = 0 ; j < width ; j++) {
            // Compute batch_value = p(x) - p(z) / (x - z).
            EF batchValue  = EF(polynomialBatch[j * height + i]) - evaluations[j];
            batchValue *= batchingPower * inverseDenom; 
            accumulator += batchValue;
            batchingPower *= batchingChallenge; 
        }
        // Add the results to the correct element of the leaf.
        size_t rowIdx = i >> 1;
        size_t isOdd = i & 1;
        for (size_t k = 0; k < bb31_extension_t::D; k++) {
            // cuda::atomic_ref<bb31_t, cuda::thread_scope_block> atomic(leafMatrix.values[(k + bb31_extension_t::D * isOdd) * leafMatrix.height + rowIdx]);
            // atomicAdd(atomic, accumulator);
            leafMatrix.values[(k + bb31_extension_t::D * isOdd) * leafMatrix.height + rowIdx] = accumulator;
      }
        domainPoint *= domainPowerStride;
    }
}
};