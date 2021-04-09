#ifndef KERNEL_SINGLETHREAD_CUH
#define KERNEL_SINGLETHREAD_CUH

#include <cstdint>

#include "cuda_time.cuh"

template <typename T>
__global__ void kernel_singlethread(T* input, uint8_t* mask, T* output, uint64_t N)
{
    uint64_t val_idx = 0;
    for (int i = 0; i < N/8; i++) {
        uint32_t acc = reinterpret_cast<uint8_t*>(mask)[i];
        for (int j = 7; j >= 0; j--) {
            uint64_t idx = i*8 + (7-j);
            bool v = 0b1 & (acc>>j);
            if (v) {
                output[val_idx++] = input[idx];
            }
        }
    }
}

template <typename T>
float launch_singlethread(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    T* d_input,
    uint8_t* d_mask,
    T* d_output,
    uint64_t N)
{
    float time;
    CUDA_TIME(ce_start, ce_stop, 0, &time,
        (kernel_singlethread<<<1,1>>>(d_input, d_mask, d_output, N))
    );
    return time;
}

#endif
