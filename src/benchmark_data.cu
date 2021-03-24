#include <assert.h>
#include <cstdint>
#include <memory>

#include "benchmark_data.cuh"
#include "cuda_try.cuh"
#include "fast_prng.hpp"

benchmark_data::benchmark_data(bool validation, uint64_t size):
    validation(validation),
    size(size)
{
    //TODO force size to multiple of 32 (datatype) and fill with 0s at end
    // alloc memory for all pointers
    uint64_t byte_size = size * sizeof(uint64_t);
    CUDA_TRY(cudaMallocHost(&h_input, byte_size));
    CUDA_TRY(cudaMallocHost(&h_mask, size / 8));
    CUDA_TRY(cudaMallocHost(&h_validation, byte_size));
    CUDA_TRY(cudaMallocHost(&h_output, byte_size));
    CUDA_TRY(cudaMalloc(&d_input, byte_size));
    CUDA_TRY(cudaMalloc(&d_mask, size / 8));
    CUDA_TRY(cudaMalloc(&d_output, byte_size));
    CUDA_TRY(cudaEventCreate(&ce_start));
    CUDA_TRY(cudaEventCreate(&ce_stop));
    // generate input
    fast_prng rng(17);
    for (int i = 0; i < size*2; i++) {
        reinterpret_cast<uint32_t*>(h_input)[i] = rng.rand();
    }
    // copy input to device
    CUDA_TRY(cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice));
}

benchmark_data::~benchmark_data()
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

void benchmark_data::generate_mask(MaskType mtype, double marg)
{
    fast_prng rng(42);
    switch (mtype)
    {
    default:
        break;
    case MASKTYPE_UNIFORM:
        // marg specifies chance of a bit being a 1
        for (int i = 0; i < size/8; i++) {
            uint32_t acc = 0;
            for (int j = 7; j >= 0; j--) {
                double r = static_cast<double>(rng.rand())/static_cast<double>(UINT32_MAX);
                if (r > marg) {
                    acc |= (1<<j);
                }
            }
            reinterpret_cast<uint8_t*>(h_mask)[i] = acc;
        }
        break;
    case MASKTYPE_ZIPF:
        // probably r = a * (c * x)^-k
        // empirical:
        // a = 1.2
        // c = log10(n) / n
        // k = 1.43
        break;
    case MASKTYPE_BURST:
        break;
    case MASKTYPE_OFFSET:
        // marg denotes that every bit at index n%marg==0 is 1 and others 0, inverted mask if marg<0
        bool invert = marg < 0;
        int64_t offset = static_cast<int64_t>(marg);
        offset = (offset == 0) ? 1 : offset;
        for (int i = 0; i < size/8; i++) {
            uint32_t acc = 0;
            for (int j = 7; j >= 0; j--) {
                if ((i*8+(7-j)) % offset == 0) {
                    acc |= (1<<j);
                }
            }
            reinterpret_cast<uint8_t*>(h_mask)[i] = (invert ? ~acc : acc);
        }
        break;
    }
    // copy mask to device
    CUDA_TRY(cudaMemcpy(d_mask, h_mask, size / 8, cudaMemcpyHostToDevice));
    // generate validation
    if (!validation) {
        return;
    }
    uint64_t val_idx = 0;
    for (int i = 0; i < size/8; i++) {
        uint32_t acc = reinterpret_cast<uint8_t*>(h_mask)[i];
        for (int j = 7; j >= 0; j--) {
            uint64_t idx = i*8 + (7-j);
            bool v = 0b1 & (acc>>j);
            if (v) {
                h_validation[val_idx++] = h_input[idx];
            }
        }
    }
    memset(&(h_validation[val_idx]), 0x00, (size-val_idx)*sizeof(uint64_t)); // set rest of validation space to 0x00
}

bool benchmark_data::validate(uint64_t count)
{
    if (count == 0) {
        count = size;
    }
    int comp = std::memcmp(h_validation, h_output, count * sizeof(uint64_t));
    if (comp != 0) {
        fprintf(stderr, "validation failed!\n");
        assert(false);
        exit(1);
    }
    return true;
}
