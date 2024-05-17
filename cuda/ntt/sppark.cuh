#include <cuda.h>


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
       void dft(fr_t* d_inout, uint32_t lg_domain_size,
        size_t device_id = 0,
        cudaStream_t hStream = cudaStream_t(cudaStreamDefault)) {

        auto& gpu = select_gpu(device_id);

        NTT::NTT_internal(d_inout, lg_domain_size,
                         NTT::InputOutputOrder::NN, NTT::Direction::forward,
                         NTT::Type::standard, gpu);

        // TODO: figure out if needed:
        // gpu.sync();
    }

    void idft(fr_t* d_inout, uint32_t lg_domain_size,
        size_t device_id = 0,
        cudaStream_t hStream = cudaStream_t(cudaStreamDefault)) {

        auto& gpu = select_gpu(device_id);

        NTT::NTT_internal(d_inout, lg_domain_size,
                         NTT::InputOutputOrder::NN, NTT::Direction::inverse,
                         NTT::Type::standard, gpu);

        // TODO: figure out if needed:
        // gpu.sync();
    }

    void coset_lde(fr_t* d_inout, uint32_t lg_domain_size, uint32_t lg_blowup,
             size_t device_id = 0,
             cudaStream_t hStream = cudaStream_t(cudaStreamDefault)) {
        size_t domain_size = (size_t)1 << lg_domain_size;
        size_t ext_domain_size = domain_size << lg_blowup;

        auto& gpu = select_gpu(device_id);

        fr_t* domain_data = &d_inout[ext_domain_size - domain_size]; // aligned to the end
        fr_t* ext_domain_data = &d_inout[0];

        NTT::NTT_internal(domain_data, lg_domain_size,
                         NTT::InputOutputOrder::NR, NTT::Direction::inverse,
                         NTT::Type::standard, gpu);

        const auto gen_powers =
                NTTParameters::all()[gpu.id()].partial_group_gen_powers;

        NTT::LDE_launch(gpu, ext_domain_data, domain_data, gen_powers,
                       lg_domain_size, lg_blowup);

        // NTT - RN
        NTT::NTT_internal(ext_domain_data, lg_domain_size + lg_blowup,
                         NTT::InputOutputOrder::RN, NTT::Direction::forward,
                         NTT::Type::standard, gpu);
        // TODO: figure out if needed:
        // gpu.sync();
    }
};

