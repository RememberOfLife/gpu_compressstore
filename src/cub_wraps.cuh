#ifndef KERNEL_4PASS_CUH
#define KERNEL_4PASS_CUH

#include <cstdint>

#include "cuda_time.cuh"
#include "cuda_try.cuh"
#include "kernels/kernel_4pass.cuh"

float launch_cub_pss(cudaEvent_t ce_start, cudaEvent_t ce_stop, uint32_t* d_pss, uint32_t* d_pss_total, uint32_t chunk_count)
{
    // use cub pss for now
    launch_4pass_pssskip(d_pss, d_pss_total, chunk_count);
    uint32_t* d_pss_tmp;
    CUDA_TRY(cudaMalloc(&d_pss_tmp, chunk_count*sizeof(uint32_t)));
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CUDA_TRY(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_pss, d_pss_tmp, chunk_count));
    CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // timed relevant computation
    float time;
    CUDA_TIME(ce_start, ce_stop, 0, &time,
        (CUDA_TRY(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_pss, d_pss_tmp, chunk_count)))
    );
    CUDA_TRY(cudaFree(d_temp_storage));
    uint32_t* d_pss_die = d_pss;
    d_pss = d_pss_tmp;
    CUDA_TRY(cudaFree(d_pss_die));
    launch_4pass_pssskip(d_pss, d_pss_total, chunk_count);
    return time;
}

#endif
