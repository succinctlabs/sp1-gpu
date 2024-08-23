
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

void dump_stage(const gpu_t& gpu, fr_t* d_inout, fr_t* h_inout, const char* stage, size_t ext_domain_size){
  size_t size = ext_domain_size * sizeof(fr_t);
  cudaMemcpyAsync(h_inout, d_inout, size, cudaMemcpyDeviceToHost, gpu);
  cudaStreamSynchronize(gpu);
  char filename[512];
  snprintf(filename, sizeof(filename), "/home/ubuntu/dev/moongate/reports/LDE-todelete/%zu_%s.txt",ext_domain_size, stage);
  FILE* file = fopen(filename, "w");
  for (size_t i = 0; i < ext_domain_size; i++) {
    fprintf(file, "%u\n", h_inout[i]);
  }
  fclose(file);
}

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
  printf("lg_domain_size: %u\n", lg_domain_size);
  printf("lg_blowup: %u\n", lg_blowup);
  printf("poly_count: %u\n", poly_count);
  printf("lg_domain_size: %u\n", lg_domain_size);

  uint32_t domain_size = 1U << lg_domain_size;
  uint32_t ext_domain_size = domain_size << lg_blowup;

  printf("domain_size: %u\n", domain_size);
  printf("ext_domain_size: %u\n", ext_domain_size);

  const gpu_t& gpu = select_gpu();

  try {
    CUDA_UNWRAP(cudaDeviceSynchronize());
  //INPUT
    for (size_t c = 0; c < poly_count; c++) {
      
      fr_t* h_inout = (fr_t*)malloc(ext_domain_size * sizeof(fr_t));
      dump_stage(gpu, d_inout, h_inout, "1", ext_domain_size);
    

      NTT::Base_dev_ptr(gpu,
                        &d_inout[(c+1) * ext_domain_size - domain_size],
                        lg_domain_size,
                        NTT::InputOutputOrder::NR,
                        NTT::Direction::inverse,
                        NTT::Type::standard);
      
      dump_stage(gpu, d_inout, h_inout, "2", ext_domain_size);

      const auto gen_powers =
          NTTParameters::all()[gpu.id()].partial_group_gen_powers;
           
      NTT::LDE_launch(gpu, 
                      &d_inout[c * ext_domain_size], 
                      &d_inout[(c + 1) * ext_domain_size - domain_size], 
                      gen_powers, 
                      lg_domain_size, 
                      lg_blowup, 
                      true, 
                      shift);

      dump_stage(gpu, d_inout, h_inout, "3", ext_domain_size);

      NTT::Base_dev_ptr(gpu,
                        &d_inout[c * ext_domain_size],
                        lg_domain_size + lg_blowup,
                        NTT::InputOutputOrder::RN,
                        NTT::Direction::forward,
                        NTT::Type::standard);

      dump_stage(gpu, d_inout, h_inout, "4", ext_domain_size);
      free(h_inout);

      if (bit_rev_output) {
           NTT::bit_rev(&d_inout[c * ext_domain_size], &d_inout[c * ext_domain_size], 
               lg_domain_size + lg_blowup, gpu);
      }
    }

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