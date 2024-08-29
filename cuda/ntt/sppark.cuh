
#include <cuda.h>

#if defined(FEATURE_BLS12_381)
# include <ff/bls12-381.hpp>
#elif defined(FEATURE_BLS12_377)
# include <ff/bls12-377.hpp>
#elif defined(FEATURE_PALLAS)
# include <ff/pasta.hpp>
#elif defined(FEATURE_VESTA)
# include <ff/pasta.hpp>
#elif defined(FEATURE_BN254)
# include <ff/alt_bn128.hpp>
#elif defined(FEATURE_GOLDILOCKS)
# include <ff/goldilocks.hpp>
#elif defined(FEATURE_BABY_BEAR)
# include <ff/baby_bear.hpp>
#else
# error "no FEATURE"
#endif

#include <ntt/ntt.cuh>

#ifndef __CUDA_ARCH__


extern "C" rustCudaError_t sppark_init() {
  uint32_t lg_domain_size = 1;
  uint32_t domain_size = 1U << lg_domain_size;

  std::vector<fr_t> inout{domain_size};
  inout[0] = fr_t(1);
  inout[1] = fr_t(1);

  const gpu_t& gpu = select_gpu();

  try {
    CUDA_UNWRAP(cudaDeviceSynchronize());

    NTT::Base(gpu,
              &inout[0],
              lg_domain_size,
              NTT::InputOutputOrder::NR,
              NTT::Direction::forward,
              NTT::Type::standard);
    gpu.sync();
  } catch (const cudaError_t& e) {
    gpu.sync();
    CUDA_OK(e);
  }
  return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t batch_lde_shift(
  fr_t* d_inout, uint32_t lg_domain_size, uint32_t lg_blowup, 
  fr_t shift, uint32_t poly_count, bool bit_rev_output) {
    
  if (lg_domain_size == 0) {
    return CUDA_SUCCESS_MOON;
  }

  uint32_t domain_size = 1U << lg_domain_size;
  uint32_t ext_domain_size = domain_size << lg_blowup;

  const gpu_t& gpu = select_gpu();

  try {
    CUDA_UNWRAP(cudaDeviceSynchronize());
    //1
    #if 0
    for (size_t c = 0; c < poly_count; c++) {
      NTT::Base_dev_ptr(gpu,  // GS_NTT
                        &d_inout[(c+1) * ext_domain_size - domain_size],
                        lg_domain_size,
                        NTT::InputOutputOrder::NR,
                        NTT::Direction::inverse,
                        NTT::Type::standard,
                        1);
    }
    #else
      NTT::Base_dev_ptr(gpu,  // GS_NTT
                        &d_inout[ext_domain_size - domain_size],
                        lg_domain_size,
                        NTT::InputOutputOrder::NR,
                        NTT::Direction::inverse,
                        NTT::Type::standard,
                        poly_count);
    #endif

    const auto gen_powers =
        NTTParameters::all()[gpu.id()].partial_group_gen_powers;
    
    //2
    #if 0
      for (size_t c = 0; c < poly_count; c++) {
        NTT::LDE_launch(gpu, 
              &d_inout[c * ext_domain_size], 
              &d_inout[(c + 1) * ext_domain_size - domain_size], 
              gen_powers, 
              lg_domain_size, 
              lg_blowup, 
              true, 
              shift,
              1);
      }
    #else
      NTT::LDE_launch(gpu, 
            &d_inout[0], 
            &d_inout[ext_domain_size - domain_size], 
            gen_powers, 
            lg_domain_size, 
            lg_blowup, 
            true, 
            shift,
            poly_count);
    #endif
    
    // dump_stage(gpu, d_inout, h_inout, "k2", 64, 4, 11);

    //3
    #if 0
    for (size_t c = 0; c < poly_count; c++) {
      NTT::Base_dev_ptr(gpu,  // CT_NTT
                        &d_inout[c * ext_domain_size],
                        lg_domain_size + lg_blowup,
                        NTT::InputOutputOrder::RN,
                        NTT::Direction::forward,
                        NTT::Type::standard,
                        1);
    }
    #else
      NTT::Base_dev_ptr(gpu,  // CT_NTT
                        &d_inout[0],
                        lg_domain_size + lg_blowup,
                        NTT::InputOutputOrder::RN,
                        NTT::Direction::forward,
                        NTT::Type::standard,
                        poly_count);
    #endif

    //4
    #if 0
      for (size_t c = 0; c < poly_count; c++) {
        if (bit_rev_output) {
           NTT::bit_rev(&d_inout[c * ext_domain_size], &d_inout[c * ext_domain_size], 
               lg_domain_size + lg_blowup, gpu, 1);
        }
      }
    #else
      if (bit_rev_output) {
          NTT::bit_rev(&d_inout[0], &d_inout[0], 
              lg_domain_size + lg_blowup, gpu, poly_count);
      }
    #endif

    gpu.sync();
  } catch (const cudaError_t& e) {
    gpu.sync();
    CUDA_OK(e);
  }

  return CUDA_SUCCESS_MOON;

}

extern "C" rustCudaError_t batch_NTT(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return CUDA_SUCCESS_MOON;

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu();

  try {
    CUDA_UNWRAP(cudaDeviceSynchronize());
    //INPUT!
    for (size_t c = 0; c < poly_count; c++) {
      NTT::Base_dev_ptr(gpu,
                        &d_inout[c * domain_size],
                        lg_domain_size,
                        NTT::InputOutputOrder::NN,
                        NTT::Direction::forward,
                        NTT::Type::standard);
    }

    gpu.sync();
    //OUTPUT!
  } catch (const cudaError_t& e) {
    gpu.sync();
    CUDA_OK(e);
  }
  return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t reverse_bits_batch(fr_t* d_out, fr_t* d_in, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return CUDA_SUCCESS_MOON;

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu();

  try {
    CUDA_UNWRAP(cudaDeviceSynchronize());

    for (size_t c = 0; c < poly_count; c++) {
      NTT::bit_rev(&d_out[c * domain_size], &d_in[c * domain_size], lg_domain_size, gpu);
    }

    gpu.sync();
  } catch (const cudaError_t& e) {
    gpu.sync();
    CUDA_OK(e);
  }
  return CUDA_SUCCESS_MOON;
}

extern "C" rustCudaError_t batch_iNTT(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return CUDA_SUCCESS_MOON;

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu();

  try {
    CUDA_UNWRAP(cudaDeviceSynchronize());
      //INPUT!
    for (size_t c = 0; c < poly_count; c++) {
      NTT::Base_dev_ptr(gpu,
                        &d_inout[c * domain_size],
                        lg_domain_size,
                        NTT::InputOutputOrder::NN,
                        NTT::Direction::inverse,
                        NTT::Type::standard);
    }

    gpu.sync();
        //OUTPUT!
  } catch (const cudaError_t& e) {
    gpu.sync();
    CUDA_OK(e);
  }
  return CUDA_SUCCESS_MOON;
}

#endif