#ifndef STREAMING_ADD_CUH
#define STREAMING_ADD_CUH

#include <cstdint>

#include "cuda_time.cuh"

__global__ void kernel_streaming_add_pss_totals(uint32_t* previous_pss_total, uint32_t* target_pss_total)
{
    *target_pss_total += *previous_pss_total;
}

void launch_streaming_add_pss_totals(cudaStream_t stream, uint32_t* d_previous_pss_total, uint32_t* d_target_pss_total)
{
    kernel_streaming_add_pss_totals<<<1,1,0,stream>>>(d_previous_pss_total, d_target_pss_total);
}

#endif
