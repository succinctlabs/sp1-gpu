#include "bb31_t.cuh"

// Pointwise summation kernel.
//
//  Each thread performs one pair-wise addition
__global__ void vecAddKernel(bb31_t *a, bb31_t *b, bb31_t *c, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}
