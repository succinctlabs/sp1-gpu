#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "../cuda/fields/bb31_t.cuh"
#include "../cuda/fields/bb31_extension_t.cuh"
#include "mersenne31/mer31_t.cuh"
#include "mersenne31/mer31_extension_t.cuh"
#include "binius/binius_t.cuh"

#define ONLY_TEST false

template <typename T>
__global__ void test_sum(T* input0, T* input1, T* output)
{
    size_t tIdx = threadIdx.x + blockDim.x * blockIdx.x;

    T value0 = input0[tIdx];
    T value1 = input1[tIdx];
    output[tIdx] = value0 + value1;
}

template <typename T>
__global__ void test_mul(T* input0, T* input1, T* output)
{
    size_t tIdx = threadIdx.x + blockDim.x * blockIdx.x;

    T value0 = input0[tIdx];
    T value1 = input1[tIdx];
    output[tIdx] = value0 * value1;
}

template <typename T>
__global__ void test_pow7(T* input0, T* output)
{
    size_t tIdx = threadIdx.x + blockDim.x * blockIdx.x;

    T value0 = input0[tIdx];
    output[tIdx] = value0^7;
}

template <typename T>
__global__ void test_reciprocal(T* input0, T* output)
{
    size_t tIdx = threadIdx.x + blockDim.x * blockIdx.x;

    T value0 = input0[tIdx];
    output[tIdx] = value0.reciprocal();
}

template <typename T>
__global__ void fill_base(uint32_t* in, T* out)
{
    size_t tIdx = threadIdx.x + blockDim.x * blockIdx.x;
    out[tIdx] = T(in[tIdx]);
}

__global__ void fill_bb31ext(bb31_t* in, bb31_extension_t* out)
{
    size_t tIdx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t N = gridDim.x * blockDim.x;

    out[tIdx] = bb31_extension_t(in[tIdx], in[N+tIdx], in[2*N+tIdx], in[3*N+tIdx]);
}

template <typename I, typename O>
__global__ void fill_ext(I* in, O* out)
{
    size_t tIdx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t N = gridDim.x * blockDim.x;

    out[tIdx] = O(in[tIdx],in[N+tIdx]);
}


class do_mod {
public:
    uint32_t val;
    static const uint32_t MOD = 0x78000001u;

    __device__ __forceinline__ do_mod(const uint32_t v) : val(((uint64_t)v << 32) % MOD) {}
};

class undo_mod {
public:
    uint32_t val;
    static const uint32_t MOD = 0x78000001u;

    __device__ __forceinline__ undo_mod(const uint32_t v) : val(v) {}
};


namespace field_test_ariphmetic {
    template <typename T>
    void test_ariphmetic (
        T* input0,
        T* input1,
        T* output,
        size_t N
    ) {
        size_t blockSize = 1024;
        size_t gridSize = N / blockSize;

        test_sum<T><<<gridSize, blockSize>>>(input0, input1, output);
        test_mul<T><<<gridSize, blockSize>>>(input0, input1, output);
        //test_pow7<T><<<gridSize, blockSize>>>(input0, output);
        //test_reciprocal<T><<<gridSize, blockSize>>>(input0, output);
    }

    void run_test(const size_t N)
    {
        std::vector<uint32_t> input;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);
        auto start = std::chrono::high_resolution_clock::now();
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> dur;

        for (int i = 0; i < N*8; i++) {
            uint32_t value = dis(gen);
            input.push_back(value);
        }

        size_t size = sizeof(uint32_t) * input.size();
        std::cout << "\n----IN TOTAL USING " << 4 * size / 1024 / 1024 << " MB----\n" << std::endl;

        uint32_t* d_input;
        cudaMalloc((void**)&d_input, size);
        cudaMemcpy(d_input, input.data(), size, cudaMemcpyHostToDevice);
        if (1) {
            bb31_t* bb31;
            cudaMalloc((void**)&bb31, size);
            if (!ONLY_TEST)
                start = std::chrono::high_resolution_clock::now();
            fill_base<bb31_t><<<N/128, 1024>>>(d_input, bb31); //KERNEL
            if (!ONLY_TEST){
                cudaDeviceSynchronize();//added
                stop = std::chrono::high_resolution_clock::now();
                dur = stop - start;
                printf("bb31 fill b     : %.2f\n", dur.count());
            } 
            bb31_t* bb31_out;            
            cudaMalloc((void**)&bb31_out, size);

            cudaDeviceSynchronize();//added
            start = std::chrono::high_resolution_clock::now();
            test_ariphmetic<bb31_t>(bb31, bb31 + N, bb31_out, N); //KERNEL
            cudaDeviceSynchronize();//added
			stop = std::chrono::high_resolution_clock::now();
			dur = stop - start;
            printf("bb31 test       : %.2f\n", dur.count());

            bb31_extension_t* bb31ext = (bb31_extension_t*)bb31_out;
            if (!ONLY_TEST)
                start = std::chrono::high_resolution_clock::now();
            fill_bb31ext<<<N/512, 1024>>>(bb31, bb31ext); //KERNEL
            if (!ONLY_TEST){ 
                cudaDeviceSynchronize();//added
                stop = std::chrono::high_resolution_clock::now();
                dur = stop - start;
                printf("bb31 ext fill b : %.2f\n", dur.count());
            }

            bb31_extension_t* bb31ext_out = (bb31_extension_t*)bb31;
            
            cudaDeviceSynchronize();//added
            start = std::chrono::high_resolution_clock::now();
            test_ariphmetic<bb31_extension_t>(bb31ext, bb31ext + N, bb31ext_out, N); //KERNEL 
            cudaDeviceSynchronize();
            stop = std::chrono::high_resolution_clock::now();
			dur = stop - start;
            printf("bb31 ext test   : %.2f\n", dur.count());

            cudaFree(bb31);
            cudaFree(bb31_out);
        }
        if (1) {
            bin32_t* bin32;
            cudaMalloc((void**)&bin32, size);
            if (!ONLY_TEST)
                start = std::chrono::high_resolution_clock::now();
            fill_base<bin32_t><<<N/128, 1024>>>(d_input, bin32); //KERNEL
            if (!ONLY_TEST){ 
                cudaDeviceSynchronize();//added
	    		stop = std::chrono::high_resolution_clock::now();
		    	dur = stop - start;
                printf("bin32 fill b    : %.2f\n", dur.count());
            }

            bin32_t* bin32_out;
            cudaMalloc((void**)&bin32_out, size);

            cudaDeviceSynchronize();//added
            start = std::chrono::high_resolution_clock::now();
            test_ariphmetic<bin32_t>(bin32, bin32 + N, bin32_out, N); //KERNEL  
            cudaDeviceSynchronize();
            stop = std::chrono::high_resolution_clock::now();
			dur = stop - start;
            printf("bin32 test      : %.2f\n", dur.count());

            bin64_t* bin64;
            cudaMalloc((void**)&bin64, sizeof(bin64_t) * input.size());
            if (!ONLY_TEST)
                start = std::chrono::high_resolution_clock::now();
            fill_ext<bin32_t, bin64_t><<<N/256, 1024>>>(bin32, bin64); //KERNEL
            if (!ONLY_TEST){ 
            cudaDeviceSynchronize();//added
                stop = std::chrono::high_resolution_clock::now();
                dur = stop - start;
                printf("bin64 fill b    : %.2f\n", dur.count());
            }

            bin64_t* bin64_out;
            cudaMalloc((void**)&bin64_out, sizeof(bin64_t) * input.size()/2);
            cudaDeviceSynchronize();//added
            start = std::chrono::high_resolution_clock::now();
            test_ariphmetic<bin64_t>(bin64, bin64 + N, bin64_out, N); //KERNEL  
            cudaDeviceSynchronize(); 
            stop = std::chrono::high_resolution_clock::now();
			dur = stop - start;
            printf("bin64 test      : %.2f\n", dur.count());

            bin128_t* bin128;
            cudaMalloc((void**)&bin128, sizeof(bin128_t) * input.size()/2);
            if (!ONLY_TEST)
                start = std::chrono::high_resolution_clock::now();
            fill_ext<bin64_t, bin128_t><<<N/512, 1024>>>(bin64, bin128); //KERNEL
            if (!ONLY_TEST){ 
            cudaDeviceSynchronize();//added
                stop = std::chrono::high_resolution_clock::now();
                dur = stop - start;
                printf("bin128 fill b   : %.2f\n", dur.count());
            }

            bin128_t* bin128_out;
            cudaMalloc((void**)&bin128_out, sizeof(bin128_t) * input.size()/4);
            cudaDeviceSynchronize();//added
            start = std::chrono::high_resolution_clock::now();
            test_ariphmetic<bin128_t>(bin128, bin128 + N, bin128_out, N); //KERNEL  
            cudaDeviceSynchronize(); 
            stop = std::chrono::high_resolution_clock::now();
			dur = stop - start;
            printf("bin128 test     : %.2f\n", dur.count());

            cudaFree(bin32);
            cudaFree(bin32_out);
            cudaFree(bin64);
            cudaFree(bin64_out);    
            cudaFree(bin128);
            cudaFree(bin128_out);
        }    
        if (1) {
            mer31_t* mer31;
            cudaMalloc((void**)&mer31, size);
            if (!ONLY_TEST)
                start = std::chrono::high_resolution_clock::now();
            fill_base<mer31_t><<<N/128, 1024>>>(d_input, mer31); //KERNEL
            if (!ONLY_TEST){ 
                cudaDeviceSynchronize();//added
	    		stop = std::chrono::high_resolution_clock::now();
		    	dur = stop - start;
                printf("mer31 fill b    : %.2f\n", dur.count());
            }

            mer31_t* mer31_out;
            cudaMalloc((void**)&mer31_out, size);

            cudaDeviceSynchronize();//added
            start = std::chrono::high_resolution_clock::now();
            test_ariphmetic<mer31_t>(mer31, mer31 + N, mer31_out, N); //KERNEL  
            cudaDeviceSynchronize();
            stop = std::chrono::high_resolution_clock::now();
			dur = stop - start;
            printf("mer31 test      : %.2f\n", dur.count());

            mer31_complex_t* merext_31;
            cudaMalloc((void**)&merext_31, sizeof(mer31_complex_t) * input.size());
            if (!ONLY_TEST)
                start = std::chrono::high_resolution_clock::now();
            fill_ext<mer31_t, mer31_complex_t><<<N/256, 1024>>>(mer31, merext_31); //KERNEL
            if (!ONLY_TEST){ 
            cudaDeviceSynchronize();//added
                stop = std::chrono::high_resolution_clock::now();
                dur = stop - start;
                printf("mer31 ext fill b: %.2f\n", dur.count());
            }

            mer31_complex_t* mer31ext_out;
            cudaMalloc((void**)&mer31ext_out, sizeof(mer31_complex_t) * input.size()/2);
            cudaDeviceSynchronize();//added
            start = std::chrono::high_resolution_clock::now();
            test_ariphmetic<mer31_complex_t>(merext_31, merext_31 + N, mer31ext_out, N); //KERNEL  
            cudaDeviceSynchronize(); 
            stop = std::chrono::high_resolution_clock::now();
			dur = stop - start;
            printf("mer31 ext test  : %.2f\n", dur.count()); 
            
            mer31_ext128_t* mer31ext_128;
            cudaMalloc((void**)&mer31ext_128, sizeof(mer31_ext128_t) * input.size()/4);
            if (!ONLY_TEST)
                start = std::chrono::high_resolution_clock::now();
            fill_ext<mer31_complex_t, mer31_ext128_t><<<N/512, 1024>>>(mer31ext_out, mer31ext_128); //KERNEL
            if (!ONLY_TEST){ 
            cudaDeviceSynchronize();//added
                stop = std::chrono::high_resolution_clock::now();
                dur = stop - start;
                printf("mer31 128 fill b: %.2f\n", dur.count());
            }

            mer31_ext128_t* mer31ext_128_out;
            cudaMalloc((void**)&mer31ext_128_out, sizeof(mer31_ext128_t) * input.size()/4);
            cudaDeviceSynchronize();//added
            start = std::chrono::high_resolution_clock::now();
            test_ariphmetic<mer31_ext128_t>(mer31ext_128, mer31ext_128 + N, mer31ext_128_out, N); //KERNEL  
            cudaDeviceSynchronize(); 
            stop = std::chrono::high_resolution_clock::now();
			dur = stop - start;
            printf("mer31 128 test  : %.2f\n", dur.count()); 

            cudaFree(mer31);
            cudaFree(mer31_out);
            cudaFree(merext_31);
            cudaFree(mer31ext_out);    
            cudaFree(mer31ext_128_out);
            cudaFree(mer31ext_128);     
        }
        if (1) {
            do_mod* bb31;
            cudaMalloc((void**)&bb31, size);
            cudaDeviceSynchronize();//added
            start = std::chrono::high_resolution_clock::now();
            fill_base<do_mod><<<N/128, 1024>>>(d_input, bb31); //KERNEL
            cudaDeviceSynchronize();     
            stop = std::chrono::high_resolution_clock::now();
			dur = stop - start;
            printf("fill base mod   : %.2f\n", dur.count()); 

            cudaFree(bb31);   
        }      
        if (1) {
            undo_mod* mer31;
            cudaMalloc((void**)&mer31, size);
            cudaDeviceSynchronize();//added
            start = std::chrono::high_resolution_clock::now();
            fill_base<undo_mod><<<N/128, 1024>>>(d_input, mer31); //KERNEL
            cudaDeviceSynchronize();     
            stop = std::chrono::high_resolution_clock::now();
			dur = stop - start;
            printf("fill base no-mod: %.2f\n", dur.count()); 

            cudaFree(mer31);     
        }
        cudaFree(d_input);
    }
}

int main()
{
    // nvcc -O3 -std=c++17 -o test_ariphmetic cuda/fields/test_ariphmetic.cu
    // ./test_ariphmetic
    field_test_ariphmetic::run_test(1024 * 1024 * 64);
    return 0;
}