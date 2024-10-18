#pragma once

#include <cuda.h>

#include <bit>

#include "../../cuda/utils/exception.cuh"

#if defined(FEATURE_BLS12_381)
    #include <ff/bls12-381.hpp>
#elif defined(FEATURE_BLS12_377)
    #include <ff/bls12-377.hpp>
#elif defined(FEATURE_PALLAS)
    #include <ff/pasta.hpp>
#elif defined(FEATURE_VESTA)
    #include <ff/pasta.hpp>
#elif defined(FEATURE_BN254)
    #include <ff/alt_bn128.hpp>
#elif defined(FEATURE_GOLDILOCKS)
    #include <ff/goldilocks.hpp>
#elif defined(FEATURE_BABY_BEAR)
    #include <ff/baby_bear.hpp>
#else
    # error "no FEATURE"
#endif

#include <ntt/ntt.cuh>

#include "moongate_cuda_cbindgen.hpp"

#ifndef __CUDA_ARCH__

namespace ntt_sppark {

inline rustCudaError_t sppark_init(const cudaStream_t stream) {
    uint32_t lg_domain_size = 1;
    uint32_t domain_size = 1U << lg_domain_size;

    std::vector<fr_t> inout {domain_size};
    inout[0] = fr_t(1);
    inout[1] = fr_t(1);
    try {
        NTT::Base(
            stream,
            &inout[0],
            lg_domain_size,
            NTT::InputOutputOrder::NR,
            NTT::Direction::forward,
            NTT::Type::standard
        );
    } catch (const cudaError_t& e) {
        CUDA_OK(e);
    }
    return CUDA_SUCCESS_MOON;
}

inline rustCudaError_t batch_lde_shift(
    fr_t* d_inout,
    uint32_t lg_domain_size,
    uint32_t lg_blowup,
    fr_t shift,
    uint32_t poly_count,
    bool bit_rev_output,
    const cudaStream_t stream
) {
    if (lg_domain_size == 0) {
        return CUDA_SUCCESS_MOON;
    }

    uint32_t domain_size = 1U << lg_domain_size;
    uint32_t ext_domain_size = domain_size << lg_blowup;

    const auto gen_powers =
        NTTParameters::all()[NTT::gpu_id()].partial_group_gen_powers;

    try {
        for (size_t c = 0; c < poly_count; c++) {
            NTT::Base_dev_ptr(
                stream,
                &d_inout[(c + 1) * ext_domain_size - domain_size],
                lg_domain_size,
                NTT::InputOutputOrder::NR,
                NTT::Direction::inverse,
                NTT::Type::standard
            );

            NTT::LDE_launch(
                stream,
                &d_inout[c * ext_domain_size],
                &d_inout[(c + 1) * ext_domain_size - domain_size],
                gen_powers,
                lg_domain_size,
                lg_blowup,
                true,
                shift
            );

            NTT::Base_dev_ptr(
                stream,
                &d_inout[c * ext_domain_size],
                lg_domain_size + lg_blowup,
                NTT::InputOutputOrder::RN,
                NTT::Direction::forward,
                NTT::Type::standard
            );

            if (bit_rev_output) {
                NTT::bit_rev(
                    &d_inout[c * ext_domain_size],
                    &d_inout[c * ext_domain_size],
                    lg_domain_size + lg_blowup,
                    stream
                );
            }
        }
    } catch (const cudaError_t& e) {
        CUDA_OK(e);
    }

    return CUDA_SUCCESS_MOON;
}

inline rustCudaError_t batch_NTT(
    fr_t* d_inout,
    uint32_t lg_domain_size,
    uint32_t poly_count,
    const cudaStream_t stream
) {
    if (lg_domain_size == 0)
        return CUDA_SUCCESS_MOON;

    uint32_t domain_size = 1U << lg_domain_size;

    try {
        for (size_t c = 0; c < poly_count; c++) {
            NTT::Base_dev_ptr(
                stream,
                &d_inout[c * domain_size],
                lg_domain_size,
                NTT::InputOutputOrder::NN,
                NTT::Direction::forward,
                NTT::Type::standard
            );
        }
    } catch (const cudaError_t& e) {
        CUDA_OK(e);
    }
    return CUDA_SUCCESS_MOON;
}

inline rustCudaError_t reverse_bits_batch(
    fr_t* d_out,
    fr_t* d_in,
    uint32_t lg_domain_size,
    uint32_t poly_count,
    const cudaStream_t stream
) {
    if (lg_domain_size == 0)
        return CUDA_SUCCESS_MOON;

    uint32_t domain_size = 1U << lg_domain_size;

    try {
        for (size_t c = 0; c < poly_count; c++) {
            NTT::bit_rev(
                &d_out[c * domain_size],
                &d_in[c * domain_size],
                lg_domain_size,
                stream
            );
        }
    } catch (const cudaError_t& e) {
        CUDA_OK(e);
    }
    return CUDA_SUCCESS_MOON;
}

inline rustCudaError_t batch_iNTT(
    fr_t* d_inout,
    uint32_t lg_domain_size,
    uint32_t poly_count,
    const cudaStream_t stream
) {
    if (lg_domain_size == 0)
        return CUDA_SUCCESS_MOON;

    uint32_t domain_size = 1U << lg_domain_size;

    try {
        for (size_t c = 0; c < poly_count; c++) {
            NTT::Base_dev_ptr(
                stream,
                &d_inout[c * domain_size],
                lg_domain_size,
                NTT::InputOutputOrder::NN,
                NTT::Direction::inverse,
                NTT::Type::standard
            );
        }
    } catch (const cudaError_t& e) {
        CUDA_OK(e);
    }
    return CUDA_SUCCESS_MOON;
}

}  // namespace ntt_sppark

namespace moongate {

CudaRustError sppark_init(CudaStreamHandle stream) {
    return ntt_sppark::sppark_init((cudaStream_t)stream);
}

CudaRustError batch_lde_shift(
    BabyBear* d_inout,
    uint32_t lg_domain_size,
    uint32_t lg_blowup,
    BabyBear shift,
    uint32_t poly_count,
    bool is_bit_rev,
    CudaStreamHandle stream
) {
    return ntt_sppark::batch_lde_shift(
        std::bit_cast<fr_t*>(d_inout),
        lg_domain_size,
        lg_blowup,
        std::bit_cast<fr_t>(shift),
        poly_count,
        is_bit_rev,
        std::bit_cast<cudaStream_t>(stream)
    );
}

CudaRustError batch_NTT(
    BabyBear* d_inout,
    uint32_t lg_domain_size,
    uint32_t poly_count,
    CudaStreamHandle stream
) {
    return ntt_sppark::batch_NTT(
        std::bit_cast<fr_t*>(d_inout),
        lg_domain_size,
        poly_count,
        std::bit_cast<cudaStream_t>(stream)
    );
}

CudaRustError batch_iNTT(
    BabyBear* d_inout,
    uint32_t lg_domain_size,
    uint32_t poly_count,
    CudaStreamHandle stream
) {
    return ntt_sppark::batch_iNTT(
        std::bit_cast<fr_t*>(d_inout),
        lg_domain_size,
        poly_count,
        std::bit_cast<cudaStream_t>(stream)
    );
}

CudaRustError reverse_bits_batch(
    BabyBear* output,
    const BabyBear* input,
    uint32_t lg_domain_size,
    uintptr_t poly_count,
    CudaStreamHandle stream
) {
    return ntt_sppark::reverse_bits_batch(
        std::bit_cast<fr_t*>(output),
        std::bit_cast<fr_t*>(input),
        lg_domain_size,
        poly_count,
        std::bit_cast<cudaStream_t>(stream)
    );
}

}  // namespace moongate

#endif