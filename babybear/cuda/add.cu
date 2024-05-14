
#include "poseidon2.cuh"
#include <cuda_runtime.h>
#include "exception.cuh"

extern "C" void poseidon2_permute_cuda(bb31_t *in, bb31_t *out, bb31_t *external_rc, bb31_t *internal_rc, int n, int numBlocks, int threadsPerBlock) {
    bb31_t *in_device;
    bb31_t *out_device;
    bb31_t *external_rc_device;
    bb31_t *internal_rc_device;
    CUDA_OK(cudaMalloc((void **)&in_device, n * WIDTH * sizeof(bb31_t)));
    CUDA_OK(cudaMalloc((void **)&out_device, n * WIDTH * sizeof(bb31_t)));
    CUDA_OK(cudaMalloc((void **)&external_rc_device, WIDTH * ROUNDS_F * sizeof(bb31_t)));
    CUDA_OK(cudaMalloc((void **)&internal_rc_device, WIDTH * ROUNDS_P * sizeof(bb31_t)));

    CUDA_OK(cudaMemcpy(in_device, in, n * WIDTH * sizeof(bb31_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(external_rc_device, external_rc, WIDTH * ROUNDS_F * sizeof(bb31_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(internal_rc_device, internal_rc, WIDTH * ROUNDS_P * sizeof(bb31_t), cudaMemcpyHostToDevice));

    poseidon2PermuteKernel<<<numBlocks, threadsPerBlock>>>(in_device, out_device, external_rc_device, internal_rc_device, n);

    CUDA_OK(cudaMemcpy(out, out_device, n * WIDTH * sizeof(bb31_t), cudaMemcpyDeviceToHost));

    CUDA_OK(cudaFree(in_device));
    CUDA_OK(cudaFree(out_device));
    CUDA_OK(cudaFree(external_rc_device));
    CUDA_OK(cudaFree(internal_rc_device));
}

extern "C" void poseidon2_external_linear_layer_cuda(bb31_t *state, int n, int numBlocks, int threadsPerBlock) {
    int size = n * 16 * sizeof(bb31_t);

    bb31_t *in;
    CUDA_OK(cudaMalloc((void **)&in, size));

    CUDA_OK(cudaMemcpy(in, state, size, cudaMemcpyHostToDevice));

    testExternalLinearLayer<<<numBlocks, threadsPerBlock>>>(in, n);

    CUDA_OK(cudaMemcpy(state, in, size, cudaMemcpyDeviceToHost));

    CUDA_OK(cudaFree(in));
}

extern "C" void poseidon2_internal_linear_layer_cuda(bb31_t *state, int n, int numBlocks, int threadsPerBlock) {
    int size = n * 16 * sizeof(bb31_t);

    bb31_t *in;
    CUDA_OK(cudaMalloc((void **)&in, size));

    CUDA_OK(cudaMemcpy(in, state, size, cudaMemcpyHostToDevice));

    testInternalLinearLayer<<<numBlocks, threadsPerBlock>>>(in, n);

    CUDA_OK(cudaMemcpy(state, in, size, cudaMemcpyDeviceToHost));

    CUDA_OK(cudaFree(in));
}