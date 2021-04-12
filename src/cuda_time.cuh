#ifndef CUDA_TIME_CUH
#define CUDA_TIME_CUH

#include <assert.h>
#include <stdio.h>

#include "cuda_try.cuh"

// macro for timing gpu operations
#define CUDA_TIME(ce_start, ce_stop, stream, time, stmt)                       \
    do {                                                                       \
        CUDA_TRY(cudaStreamSynchronize((stream)));                             \
        CUDA_TRY(cudaEventRecord((ce_start)));                                 \
        stmt;                                                                  \
        CUDA_TRY(cudaEventRecord((ce_stop)));                                  \
        CUDA_TRY(cudaEventSynchronize((ce_stop)));                             \
        CUDA_TRY(cudaEventElapsedTime((time), (ce_start), (ce_stop)));         \
    } while (0)

#endif
