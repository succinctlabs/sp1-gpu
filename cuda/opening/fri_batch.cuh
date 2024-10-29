
#pragma once

#include <ntt/ntt.cuh>
#include <cuda/atomic>

#include "../fields/bb31_extension_t.cuh"
#include "../fields/bb31_t.cuh"


namespace fri_batch {
// template <typename T>
// __device__ void atomicAdd(T* dst, const T value) {
//     T old_val = *dst;
//     T new_val;

//     do {
//         new_val = old_val + value;
//     } while (!atomic_ref.compare_exchange_weak(old_val, new_val, cuda::std::memory_order_relaxed));
// }

template<typename F, typename EF> __global__ void batchFriKernel(
    Matrix<F> leafMatrix,
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
    if (i >= height) return;

    F domainPoint = domainGenerator^(bit_rev(blockIdx.x * blockDim.x + threadIdx.x, logHeight));
    domainPoint *= shift;
    EF batchingPower = batchingChallengeOffset;
    EF inverseDenom = EF::one() / (evaluationPoint - domainPoint);
    EF accumulator = EF::zero(); 
    for (size_t j = 0 ; j < width ; j++) {
        // Compute batch_value = alpha^i ((p(z) - p(x) / (z - x)).
        EF batchValue  = evaluations[j] - polynomialBatch[j * height + i];
        batchValue *= batchingPower; 
        accumulator += batchValue;
        batchingPower *= batchingChallenge; 
    }
    accumulator *= inverseDenom;
    // Add the results to the correct element of the leaf.
    size_t rowIdx = i >> 1;
    size_t isOdd = i & 1;
    for (size_t k = 0; k < EF::D; k++) {
        leafMatrix.values[(k + isOdd *  EF::D) * leafMatrix.height + rowIdx] += accumulator.value[k];
    }
}


extern "C" void batchFri(
    Matrix<bb31_t> leafMatrix,
    const bb31_t* polynomialBatch,
    const bb31_extension_t* evaluations,
    const bb31_t domainGenerator,
    const bb31_t shift,
    const bb31_extension_t evaluationPoint,
    const bb31_extension_t batchingChallenge,
    const bb31_extension_t batchingChallengeOffset,
    size_t width,
    size_t logHeight,
    cudaStream_t stream) {
        size_t blockDim = 512;
        size_t height = 1U << logHeight;
        size_t gridDim = (height - 1) / blockDim + 1;

        batchFriKernel<<<gridDim, blockDim, 0, stream>>>(
            leafMatrix,
            polynomialBatch,
            evaluations,
            domainGenerator,
            shift,
            evaluationPoint,
            batchingChallenge,
            batchingChallengeOffset,
            width,
            logHeight);
    }
};