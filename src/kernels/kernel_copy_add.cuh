#ifndef KERNEL_COPY_CUH
#define KERNEL_COPY_CUH

#include <cstdint>

#include "cuda_time.cuh"

template <typename T>
__global__ void kernel_copy(T* input, T* output, uint64_t N)
{
    for (uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x) {
        output[tid] = input[tid];
    }
}

template <typename T>
__global__ void kernel_copy_add(T* input, T* output, uint8_t* mask, uint64_t N)
{
    for (uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x) {
        output[tid] = input[tid] + mask[tid%(N/8)];
    }
}

template <typename T>
float launch_copy_add(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    T* d_input,
    T* d_output,
    uint8_t* d_mask,
    uint64_t N,
    bool do_add)
{
    float time;
    if (blockcount == 0) {
        blockcount = N / threadcount;
    }
    if (do_add) {
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_copy<<<blockcount, threadcount>>>(d_input, d_output, N))
        );
    } else {
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_copy_add<<<blockcount, threadcount>>>(d_input, d_output, d_mask, N))
        );
    }
    return time;
}

#endif
