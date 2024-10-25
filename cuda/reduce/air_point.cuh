#pragma once

#include "reduce.cuh"

template<typename F> 
 struct air_point_t {
    F local;
    F next;

    __device__ __forceinline__ air_point_t(F local, F next) {
        this->local = local;
        this->next = next;
    }

    static __device__ __forceinline__ const air_point_t zero() {
        return air_point_t(F::zero(), F::zero());
    }

    __device__ __forceinline__ air_point_t& operator+=(const air_point_t b) {
        local += b.local;
        next += b.next;
        return *this;
    }

    friend __device__ __forceinline__ air_point_t operator+(air_point_t a,
                                             const air_point_t b) {
        return a += b;
    }

    __device__ __forceinline__ air_point_t& operator-=(const air_point_t b) {
        local-= b.local;
        next -= b.next;
        return *this;
    }

    friend __device__ __forceinline__ air_point_t operator-(air_point_t a,
                                             const air_point_t b) {
        return a -= b;
    }

    __device__ __forceinline__ air_point_t& operator*=(const F b) {
        local *= b;
        next *= b;
        return *this;
    }

    friend __device__ __forceinline__ air_point_t operator*(air_point_t a, const F b) {
        return a *= b;
    }
};



template <typename F>
  struct AddOpFinalReduce<air_point_t<F>> {
    template<typename TyGroup>
    __device__ __forceinline__ static void final_block_reduction_async(
        const TyGroup& group, 
        air_point_t<F>* dst, 
        air_point_t<F> val) {
        AddOp<F> op;
        op.final_block_reduction_async(group, &(dst->local), val.local);
        op.final_block_reduction_async(group, &(dst->next), val.next);
    }
  };


extern "C" RustCudaError vectorsumAirBabyBearExtension(
    air_point_t<bb31_extension_t>* in,
    air_point_t<bb31_extension_t>* result, 
    size_t width, size_t height, 
    cudaStream_t stream) {
    return vectorSum(in, result, width, height, stream);
}
