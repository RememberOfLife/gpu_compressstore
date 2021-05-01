#ifndef STREAMING_3PASS_CUH
#define STREAMING_3PASS_CUH

#include <chrono>
#include <cstdint>

#include "cuda_try.cuh"
#include "kernels/kernel_3pass.cuh"

template <typename T>
float launch_streaming_3pass(T* d_input, uint8_t* d_mask, T* d_output, uint64_t N, int stream_count) {
    if (stream_count < 1) {
        stream_count = 1;
    }
    // only works for power of two streams
    int p2_sc = 1;
    while (p2_sc < stream_count) {
        p2_sc *= 2;
    }
    stream_count = p2_sc;

    uint32_t chunk_length = 1024;
    uint32_t chunk_count = N / chunk_length;
    uint32_t max_chunk_count = N / 32;
    uint32_t* d_pss;
    CUDA_TRY(cudaMalloc(&d_pss, max_chunk_count*sizeof(uint32_t)));
    uint32_t* d_popc;
    CUDA_TRY(cudaMalloc(&d_popc, max_chunk_count*sizeof(uint32_t)));

    uint32_t* d_pss_total;
    CUDA_TRY(cudaMalloc(&d_pss_total, sizeof(uint32_t)));
    CUDA_TRY(cudaMemset(d_pss_total, 0x00, sizeof(uint32_t)));

    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*stream_count);
    for (int i = 0; i < stream_count; i++) {
        CUDA_TRY(cudaStreamCreate(&(streams[i])));
    }

    uint64_t elems_per_stream = N / stream_count;
    uint64_t bytes_per_stream = elems_per_stream / 8;
    uint64_t chunks_per_stream = elems_per_stream / chunk_length;
    uint64_t chunks1024_per_stream = elems_per_stream / 1024;

    uint32_t chunk_count_p2 = 1;
    while (chunk_count_p2 < chunks_per_stream) {
        chunk_count_p2 *= 2;
    }

    uint32_t chunk_length32 = chunk_length / 32;
    int popc1_threadcount = 1024;
    int popc1_blockcount = (chunk_count/popc1_threadcount)+1;
    int popc2_threadcount = 1024;
    int popc2_blockcount = (chunk_count/popc2_threadcount)+1;
    const int proc_threadcount = 1024;
    int proc_blockcount = chunk_count/1024;
    if (proc_blockcount < 1) {
        proc_blockcount = 1;
    }
    uint64_t out_count = 0;

    std::chrono::time_point<std::chrono::steady_clock> start_clock = std::chrono::steady_clock::now();
    
    for (int j = 0; j < stream_count+2; j++) {
        int i = j;
        // popc block at i
        if (i < stream_count) {
            kernel_3pass_popc_none_monolithic<<<popc1_blockcount, popc1_threadcount, 0, streams[i]>>>(d_mask+bytes_per_stream*i, d_pss+chunks_per_stream*i, chunk_length32, chunks_per_stream);
        }
        uint32_t h_pss_total = 0;
        // cub pss at i-1
        i--;
        if (i >= 0 && i < stream_count) {
            if (i > 0) {
                CUDA_TRY(cudaMemcpyAsync(&h_pss_total, d_pss_total, sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[i-1])); // copy over old d_pss_total for writeout
                CUDA_TRY(cudaStreamSynchronize(streams[i-1])); // likely unneccessary sync back to cpu
                //FIX use multiple d_pss_out and sync by wait events between the streams for easy launching
            }
            CUDA_TRY(cudaMemsetAsync(d_pss_total, 0x00, sizeof(uint32_t), streams[i]));
            launch_cub_pss(streams[i], 0, 0, d_pss+chunks_per_stream*i, d_pss_total, chunks_per_stream);
        }
        // optimized writeout at i-2
        i--;
        if (i >= 0) {
            // optimization popc for 1024bit chunks
            kernel_3pass_popc_none_monolithic<<<popc2_blockcount, popc2_threadcount, 0, streams[i]>>>(d_mask+bytes_per_stream*i, d_popc+chunks1024_per_stream*i, 1024/32, chunks1024_per_stream);
            kernel_3pass_proc_true_striding<proc_threadcount, T, true><<<proc_blockcount, proc_threadcount, 0, streams[i]>>>(d_input+elems_per_stream*i, d_output+out_count, d_mask+bytes_per_stream*i, d_pss+chunks_per_stream*i, d_popc+chunks1024_per_stream*i, chunk_length, chunks_per_stream, chunk_count_p2);
        }
        out_count += h_pss_total;
    }
    CUDA_TRY(cudaDeviceSynchronize());
    std::chrono::time_point<std::chrono::steady_clock> stop_clock = std::chrono::steady_clock::now();
    
    for (int i = 0; i < stream_count; i++) {
        CUDA_TRY(cudaStreamDestroy(streams[i]));
    }
    free(streams);

    return static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop_clock-start_clock).count()) / 1000000;
}

#endif
