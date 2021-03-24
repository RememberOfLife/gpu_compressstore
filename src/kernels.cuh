#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cstdint>

#include "kernels.cuh"

__global__ void kernel_singlethread(uint64_t* input, uint8_t* mask, uint64_t* output, uint64_t N);

#endif
