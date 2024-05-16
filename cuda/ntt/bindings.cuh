

#include "../fields/bb31_t.cuh"

#ifdef SPPARK

#include "sppark.cuh"
extern "C" namespace ntt {
    void coset_lde(bb31_t* d_inout, uint32_t lg_domain_size, uint32_t lg_blowup) {
        DftSppark sspark;
        sspark.coset_lde(d_inout, lg_domain_size, lg_blowup);
    }
}

#else

#error "not supported" 

#endif