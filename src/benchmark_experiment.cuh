#ifndef BENCHMARK_EXPERIMENT_CUH
#define BENCHMARK_EXPERIMENT_CUH

#include <cstdint>
#include <fstream>
#include <iostream>

#include "benchmark_data.cuh"
#include "cub_wraps.cuh"
#include "cuda_try.cuh"
#include "kernels/kernel_4pass.cuh"
#include "kernels/kernel_singlethread.cuh"

#define RUNS_WARMUP 2
#define RUNS_MEASURE 20

template <typename T>
void run_benchmark_experiment(benchmark_data<T>* bdata) {
    std::ofstream result_data;
    result_data.open("result_data.csv");
    if (!result_data) {
        std::cerr << "error: result file could not be opened\n";
        exit(1);
    }
    cudaDeviceProp deviceProp;
    CUDA_TRY(cudaGetDeviceProperties(&deviceProp, 0));
    uint32_t sm_count = deviceProp.multiProcessorCount;
    // actual benchmarking begins
    result_data << "datasize;algo;chunklength;blocks;threads;time\n";
    float timer;

    uint32_t* d_pss_total; // pss total
    CUDA_TRY(cudaMalloc(&d_pss_total, sizeof(uint32_t)));
    CUDA_TRY(cudaMemset(d_pss_total, 0x00, sizeof(uint32_t)));
    uint32_t h_pss_total = 256;//33555616; //FIXME get true value from bdata

    //BENCHMARK single_thread
    launch_singlethread(bdata->ce_start, bdata->ce_stop, bdata->d_input, bdata->d_mask, bdata->d_output, bdata->count);
    for (int r = 0; r < 3; r++) {
        timer = launch_singlethread(bdata->ce_start, bdata->ce_stop, bdata->d_input, bdata->d_mask, bdata->d_output, bdata->count);
        CUDA_TRY(cudaMemcpy(bdata->h_output, bdata->d_output, bdata->count*sizeof(T), cudaMemcpyDeviceToHost));
        if (!bdata->validate(h_pss_total)) { timer = -1; }
        result_data << (bdata->count*sizeof(T)) << ";single_thread;0;1;1;" << timer << "\n";
    }

    //BENCHMARK cub_flagged_bytemask
    for (int r = 0; r < RUNS_WARMUP; r++) {
        launch_cub_flagged_bytemask(bdata->ce_start, bdata->ce_stop, bdata->d_input, bdata->d_output, bdata->h_mask, d_pss_total, bdata->count);
    }
    for (int r = 0; r < RUNS_MEASURE; r++) {
        timer = launch_cub_flagged_bytemask(bdata->ce_start, bdata->ce_stop, bdata->d_input, bdata->d_output, bdata->h_mask, d_pss_total, bdata->count);
        CUDA_TRY(cudaMemcpy(bdata->h_output, bdata->d_output, bdata->count*sizeof(T), cudaMemcpyDeviceToHost));
        if (!bdata->validate(h_pss_total)) { timer = -1; }
        result_data << (bdata->count*sizeof(T)) << ";cub_flagged_bytemask;0;0;0;" << timer << "\n";
    }

    //BENCHMARK cub_flagged_biterator
    for (int r = 0; r < RUNS_WARMUP; r++) {
        launch_cub_flagged_biterator(bdata->ce_start, bdata->ce_stop, bdata->d_input, bdata->d_output, bdata->d_mask, d_pss_total, bdata->count);
    }
    for (int r = 0; r < RUNS_MEASURE; r++) {
        timer = launch_cub_flagged_biterator(bdata->ce_start, bdata->ce_stop, bdata->d_input, bdata->d_output, bdata->d_mask, d_pss_total, bdata->count);
        CUDA_TRY(cudaMemcpy(bdata->h_output, bdata->d_output, bdata->count*sizeof(T), cudaMemcpyDeviceToHost));
        if (!bdata->validate(h_pss_total)) { timer = -1; }
        result_data << (bdata->count*sizeof(T)) << ";cub_flagged_biterator;0;0;0;" << timer << "\n";
    }
    
    //BENCHMARK 3pass
    for (int chunk_length = 32; chunk_length < 1024; chunk_length *= 2) {
        uint32_t chunk_count = bdata->count / chunk_length;
        uint32_t* d_pss; // prefix sum scan buffer on device
        CUDA_TRY(cudaMalloc(&d_pss, chunk_count*sizeof(uint32_t)));
        uint32_t* d_pss_full; // prefix sum scan buffer on device
        CUDA_TRY(cudaMalloc(&d_pss_full, chunk_count*sizeof(uint32_t)));
        iovRow* d_iov; // intermediate optimization vector
        CUDA_TRY(cudaMalloc(&d_iov, chunk_count*sizeof(iovRow)));

        //BENCHMARK cub_pss
        for (int r = 0; r < RUNS_WARMUP; r++) {
            launch_4pass_popc_none(bdata->ce_start, bdata->ce_stop, 0, 256, bdata->d_mask, d_pss_full, d_iov, chunk_length, chunk_count);
            launch_cub_pss(bdata->ce_start, bdata->ce_stop, d_pss_full, d_pss_total, chunk_count);
        }
        for (int r = 0; r < RUNS_MEASURE; r++) {
            launch_4pass_popc_none(bdata->ce_start, bdata->ce_stop, 0, 256, bdata->d_mask, d_pss_full, d_iov, chunk_length, chunk_count);
            timer = launch_cub_pss(bdata->ce_start, bdata->ce_stop, d_pss_full, d_pss_total, chunk_count);
            CUDA_TRY(cudaMemcpy(bdata->h_output, bdata->d_output, bdata->count*sizeof(T), cudaMemcpyDeviceToHost));
            if (!bdata->validate(h_pss_total)) { timer = -1; }
            result_data << (bdata->count*sizeof(T)) << ";cub_pss;" << chunk_length << ";0;0;" << timer << "\n";
        }

        
        //BENCHMARK 3pass popc
        for (int block_count = 0; block_count < 100; block_count += sm_count) {
            for (int thread_count = 32; thread_count < 1024; thread_count *= 2) {

                //BECHMARK 3pass_popc_none
                for (int r = 0; r < RUNS_WARMUP; r++) {
                    launch_4pass_popc_none(bdata->ce_start, bdata->ce_stop, block_count, thread_count, bdata->d_mask, d_pss, d_iov, chunk_length, chunk_count);
                }
                for (int r = 0; r < RUNS_MEASURE; r++) {
                    timer = launch_4pass_popc_none(bdata->ce_start, bdata->ce_stop, block_count, thread_count, bdata->d_mask, d_pss, d_iov, chunk_length, chunk_count);
                    CUDA_TRY(cudaMemcpy(bdata->h_output, bdata->d_output, bdata->count*sizeof(T), cudaMemcpyDeviceToHost));
                    if (!bdata->validate(h_pss_total)) { timer = -1; }
                    result_data << (bdata->count*sizeof(T)) << ";3pass_popc_none;" << chunk_length << ";" << block_count << ";" << thread_count << ";" << timer << "\n";
                }

                //BECHMARK 3pass_pss_gmem
                for (int r = 0; r < RUNS_WARMUP; r++) {
                    launch_4pass_popc_none(bdata->ce_start, bdata->ce_stop, block_count, thread_count, bdata->d_mask, d_pss, d_iov, chunk_length, chunk_count);
                    launch_4pass_pss_gmem(bdata->ce_start, bdata->ce_stop, block_count, thread_count, d_pss, chunk_count, d_pss_total);
                }
                for (int r = 0; r < RUNS_MEASURE; r++) {
                    launch_4pass_popc_none(bdata->ce_start, bdata->ce_stop, block_count, thread_count, bdata->d_mask, d_pss, d_iov, chunk_length, chunk_count);
                    timer = launch_4pass_pss_gmem(bdata->ce_start, bdata->ce_stop, block_count, thread_count, d_pss, chunk_count, d_pss_total);
                    CUDA_TRY(cudaMemcpy(bdata->h_output, bdata->d_output, bdata->count*sizeof(T), cudaMemcpyDeviceToHost));
                    if (!bdata->validate(h_pss_total)) { timer = -1; }
                    result_data << (bdata->count*sizeof(T)) << ";3pass_pss_gmem;" << chunk_length << ";" << block_count << ";" << thread_count << ";" << timer << "\n";
                }

                //BECHMARK 3pass_proc_true PARTIAL
                for (int r = 0; r < RUNS_WARMUP; r++) {
                    launch_4pass_proc_true(bdata->ce_start, bdata->ce_stop, block_count, thread_count, bdata->d_input, bdata->d_output, bdata->d_mask, d_pss, false, chunk_length, chunk_count);
                }
                for (int r = 0; r < RUNS_MEASURE; r++) {
                    timer = launch_4pass_proc_true(bdata->ce_start, bdata->ce_stop, block_count, thread_count, bdata->d_input, bdata->d_output, bdata->d_mask, d_pss, false, chunk_length, chunk_count);
                    CUDA_TRY(cudaMemcpy(bdata->h_output, bdata->d_output, bdata->count*sizeof(T), cudaMemcpyDeviceToHost));
                    if (!bdata->validate(h_pss_total)) { timer = -1; }
                    result_data << (bdata->count*sizeof(T)) << ";3pass_pproc_true;" << chunk_length << ";" << block_count << ";" << thread_count << ";" << timer << "\n";
                }

                //BECHMARK 3pass_proc_true FULL
                for (int r = 0; r < RUNS_WARMUP; r++) {
                    launch_4pass_proc_true(bdata->ce_start, bdata->ce_stop, block_count, thread_count, bdata->d_input, bdata->d_output, bdata->d_mask, d_pss_full, true, chunk_length, chunk_count);
                }
                for (int r = 0; r < RUNS_MEASURE; r++) {
                    timer = launch_4pass_proc_true(bdata->ce_start, bdata->ce_stop, block_count, thread_count, bdata->d_input, bdata->d_output, bdata->d_mask, d_pss_full, true, chunk_length, chunk_count);
                    CUDA_TRY(cudaMemcpy(bdata->h_output, bdata->d_output, bdata->count*sizeof(T), cudaMemcpyDeviceToHost));
                    if (!bdata->validate(h_pss_total)) { timer = -1; }
                    result_data << (bdata->count*sizeof(T)) << ";3pass_fproc_true;" << chunk_length << ";" << block_count << ";" << thread_count << ";" << timer << "\n";
                }

            }
        }

        CUDA_TRY(cudaFree(d_iov));
        CUDA_TRY(cudaFree(d_pss));
    }

    CUDA_TRY(cudaFree(d_pss_total));
    // close file handler
    result_data.close();
}

#endif
