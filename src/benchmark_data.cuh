#ifndef BENCHMARK_DATA_CUH
#define BENCHMARK_DATA_CUH

#include <assert.h>
#include <cstdint>
#include <memory>

#include "benchmark_data.cuh"
#include "cuda_try.cuh"
#include "fast_prng.hpp"

enum MaskType {
    MASKTYPE_UNIFORM = 0, // every bit is decided uniformly random
    MASKTYPE_ZIPF = 1, // lots of 1s in the front thinning out quickly towards the back
    MASKTYPE_BURST = 2, // randomly placed bursts of 1s
    MASKTYPE_OFFSET = 3, // 1s with gaps of 0s: 1=101010..; 2=1001001..
};

template <typename T>
struct benchmark_data {
    uint64_t count; // elems
    T* h_input;
    uint8_t* h_mask;
    T* h_validation; // correct results
    T* h_output; // gpu result buffer
    T* d_input;
    uint8_t* d_mask;
    T* d_output;
    cudaEvent_t ce_start;
    cudaEvent_t ce_stop;

    benchmark_data(uint64_t count):
        count(count)
    {
        //TODO force byte_size to multiple of 32bit and element count to multiple of 8; fill with 0s at end
        // alloc memory for all pointers
        uint64_t byte_size_data = count * sizeof(T);
        uint64_t byte_size_mask = count / 8;
        CUDA_TRY(cudaMallocHost(&h_input, byte_size_data));
        CUDA_TRY(cudaMallocHost(&h_mask, byte_size_mask));
        CUDA_TRY(cudaMallocHost(&h_validation, byte_size_data));
        CUDA_TRY(cudaMallocHost(&h_output, byte_size_data));
#ifdef AVXPOWER
        // assert aligned malloc
        assert(h_input % sizeof(T) == 0);
        assert(h_validation % sizeof(T) == 0);
        assert(h_output % sizeof(T) == 0);
#endif
        CUDA_TRY(cudaMalloc(&d_input, byte_size_data));
        CUDA_TRY(cudaMalloc(&d_mask, byte_size_mask));
        CUDA_TRY(cudaMalloc(&d_output, byte_size_data));
        CUDA_TRY(cudaEventCreate(&ce_start));
        CUDA_TRY(cudaEventCreate(&ce_stop));
        // generate input
        fast_prng rng(17);
        for (int i = 0; i < byte_size_data/4; i++) {
            reinterpret_cast<uint32_t*>(h_input)[i] = rng.rand();
        }
        // copy input to device
        CUDA_TRY(cudaMemcpy(d_input, h_input, byte_size_data, cudaMemcpyHostToDevice));
        // clear device output
        CUDA_TRY(cudaMemset(d_output, 0x00, byte_size_data));
    }

    ~benchmark_data()
    {
        CUDA_TRY(cudaEventDestroy(ce_stop));
        CUDA_TRY(cudaEventDestroy(ce_start));
        CUDA_TRY(cudaFree(d_output));
        CUDA_TRY(cudaFree(d_mask));
        CUDA_TRY(cudaFree(d_input));
        CUDA_TRY(cudaFreeHost(h_output));
        CUDA_TRY(cudaFreeHost(h_validation));
        CUDA_TRY(cudaFreeHost(h_mask));
        CUDA_TRY(cudaFreeHost(h_input));
    }

    uint32_t generate_mask(MaskType mtype, double marg)
    {
        fast_prng rng(42);
        switch (mtype)
        {
        default:
            break;
        case MASKTYPE_UNIFORM: {
                // marg specifies chance of a bit being a 1
                for (int i = 0; i < count/8; i++) {
                    uint8_t acc = 0;
                    for (int j = 7; j >= 0; j--) {
                        if (rng.rand() < marg*UINT32_MAX) {
                            acc |= (1<<j);
                        }
                    }
                    reinterpret_cast<uint8_t*>(h_mask)[i] = acc;
                }
            }
            break;
        case MASKTYPE_ZIPF: {
                // probably r = a * (c * x)^-k
                // empirical:
                // a = 1.2
                // c = log10(n) / n
                // k = 1.43
                double c = log10(static_cast<double>(count)) / static_cast<double>(count);
                for (int i = 0; i < count/8; i++) {
                    uint8_t acc = 0;
                    for (int j = 7; j >= 0; j--) {
                        double ev = marg * (1 / (pow((c*(i*8+(7-j))), 1.43)));
                        double rv = static_cast<double>(rng.rand())/static_cast<double>(UINT32_MAX);
                        if (rv < ev) {
                            acc |= (1<<j);
                        }
                    }
                    reinterpret_cast<uint8_t*>(h_mask)[i] = acc;
                }
            }
            break;
        case MASKTYPE_BURST: {
                // marg sets pseudo segment distance, can be modified by up to +/-50% in size and is randomly 1/0
                double segment = static_cast<double>(count) * marg;
                double rv = static_cast<double>(rng.rand())/static_cast<double>(UINT32_MAX);
                uint64_t current_length = static_cast<uint64_t>(segment * (rv+0.5));
                bool is_one = false;
                for (int i = 0; i < count/8; i++) {
                    uint8_t acc = 0;
                    for (int j = 7; j >= 0; j--) {
                        if (is_one) {
                            acc |= (1<<j);
                        }
                        if (--current_length == 0) {
                            rv = static_cast<double>(rng.rand())/static_cast<double>(UINT32_MAX);
                            current_length = static_cast<uint64_t>(segment * (rv+0.5));
                            is_one = !is_one;
                        }
                    }
                    reinterpret_cast<uint8_t*>(h_mask)[i] = acc;
                }
            }
            break;
        case MASKTYPE_OFFSET: {
                // marg denotes that every bit at index n%marg==0 is 1 and others 0, inverted mask if marg<0
                bool invert = marg < 0;
                int64_t offset = static_cast<int64_t>(marg);
                offset = (offset == 0) ? 1 : offset;
                for (int i = 0; i < count/8; i++) {
                    uint8_t acc = 0;
                    for (int j = 7; j >= 0; j--) {
                        if ((i*8+(7-j)) % offset == 0) {
                            acc |= (1<<j);
                        }
                    }
                    reinterpret_cast<uint8_t*>(h_mask)[i] = (invert ? ~acc : acc);
                }
            }
            break;
        }
        // copy mask to device
        CUDA_TRY(cudaMemcpy(d_mask, h_mask, count / 8, cudaMemcpyHostToDevice));
        // generate validation
        uint32_t onecount = 0;
        uint64_t val_idx = 0;
        for (int i = 0; i < count/8; i++) {
            uint32_t acc = reinterpret_cast<uint8_t*>(h_mask)[i];
            for (int j = 7; j >= 0; j--) {
                uint64_t idx = i*8 + (7-j);
                bool v = 0b1 & (acc>>j);
                if (v) {
                    onecount++;
                    h_validation[val_idx++] = h_input[idx];
                }
            }
        }
        memset(&(h_validation[val_idx]), 0x00, (count-val_idx)*sizeof(T)); // set rest of validation space to 0x00
        return onecount;
    }

    bool validate(uint64_t count)
    {
        if (count == 0) {
            count = count;
        }
        for (uint32_t i = 0; i < count; i++) {
            if (h_validation[i] != h_output[i]) {
                for (uint32_t j = 0; j < count; j++) {
                    if (h_input[j] == h_validation[i]) {
                        fprintf(stderr, "validation failed (got %llu @ %d, expected %llu from %d)\n", h_output[i], i, h_validation[i], j);
                        break;
                    }
                }
                return false;
            }
        }
        return true;
    }
};

#endif
