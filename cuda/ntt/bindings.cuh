#include "../fields/bb31_t.cuh"


#include "sppark.cuh"
extern "C"  {
    void coset_lde(bb31_t* d_inout, uint32_t lg_domain_size, uint32_t lg_blowup) {
        DftSppark sspark;
        sspark.coset_lde(d_inout, lg_domain_size, lg_blowup);
    }

    void dft(bb31_t* d_inout, uint32_t lg_domain_size) {
        DftSppark sspark;
        sspark.dft(d_inout, lg_domain_size);
    }

    void idft(bb31_t* d_inout, uint32_t lg_domain_size) {
        DftSppark sspark;
        sspark.idft(d_inout, lg_domain_size);
    }
};
