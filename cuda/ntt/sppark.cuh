
#include <cuda.h>

#ifdef SPPARK

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
#error "no FEATURE"
#endif

#include <ntt/ntt.cuh>

class DftSppark {
   public:
    void lde(fr_t* inout, uint32_t lg_domain_size, size_t device_id = 0,
             cudaStream_t hStream = cudaStream_t(cudaStreamDefault)) {
        auto& gpu = select_gpu(device_id);
    }
};

#endif
