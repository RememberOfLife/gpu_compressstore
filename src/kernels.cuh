#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cstdint>

#include "kernels.cuh"

template <typename T>
__global__ void kernel_singlethread(T* input, uint8_t* mask, T* output, uint64_t N);

#endif
