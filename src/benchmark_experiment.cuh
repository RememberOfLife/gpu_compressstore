#ifndef BENCHMARK_EXPERIMENT_CUH
#define BENCHMARK_EXPERIMENT_CUH

#include <cstdint>
#include <fstream>
#include <iostream>

#include "avx_wrap.cuh"
#include "benchmark_data.cuh"
#include "cub_wraps.cuh"
#include "cuda_try.cuh"
#include "kernels/kernel_3pass.cuh"
#include "kernels/kernel_singlethread.cuh"

#define RUNS_MEASURE 2

// benchmark for 3pass variations
template <typename T>
void run_single_benchmark(
    benchmark_data<T>* bdata,
    std::ofstream& result_data,
    uint32_t onecount,
    uint32_t* d_pss,
    uint32_t* d_pss2,
    uint32_t* d_pss_total,
    uint32_t* d_popc,
    bool cub_pss,
    bool optimized_writeout_order,
    uint32_t chunk_length,
    uint32_t chunk_count,
    uint32_t block_count,
    uint32_t thread_count)
{
    // uses std::endl to force data into file even on early termination
    float time;
    // popcount
    time = launch_3pass_popc_none(bdata->ce_start, bdata->ce_stop, block_count, thread_count, bdata->d_mask, d_pss, chunk_length, chunk_count);
    result_data << (bdata->count * sizeof(T)) << ";" << bdata->p << ";3pass_popc_none;" << chunk_length << ";" << block_count << ";" << thread_count << ";" << time << std::endl;

    // prefix sum
    if (cub_pss) {
        time = launch_cub_pss(bdata->ce_start, bdata->ce_stop, d_pss, d_pss_total, chunk_count);
        result_data << (bdata->count * sizeof(T)) << ";" << bdata->p << ";cub_pss;" << chunk_length << ";0;0;" << time << std::endl;
    }
    else {
        time = launch_3pass_pss_gmem(bdata->ce_start, bdata->ce_stop, block_count, thread_count, d_pss, chunk_count, d_pss_total);
        result_data << (bdata->count * sizeof(T)) << ";" << bdata->p << ";3pass_pss_gmem;" << chunk_length << ";" << block_count << ";" << thread_count << ";" << time << std::endl;
        time = launch_3pass_pss2_gmem(bdata->ce_start, bdata->ce_stop, block_count, thread_count, d_pss, d_pss2, chunk_count);
        result_data << (bdata->count * sizeof(T)) << ";" << bdata->p << ";3pass_pss2_gmem;" << chunk_length << ";" << block_count << ";" << thread_count << ";" << time << std::endl;
    }

    CUDA_TRY(cudaMemset(bdata->d_output, 0x00, bdata->count)); // reset output between runs

    // writeout
    if (optimized_writeout_order) {
        time = launch_3pass_proc_true(bdata->ce_start, bdata->ce_stop, block_count, thread_count, bdata->d_input, bdata->d_output, bdata->d_mask, d_pss, cub_pss, d_popc, chunk_length, chunk_count);
        CUDA_TRY(cudaMemcpy(bdata->h_output, bdata->d_output, bdata->count*sizeof(T), cudaMemcpyDeviceToHost));
        if (!bdata->validate(onecount)) { time = -1; }
        result_data << (bdata->count * sizeof(T)) << ";" << bdata->p << ";" << (cub_pss ? "3pass_fproc_true" : "3pass_pproc_true") << ";" << chunk_length << ";" << block_count << ";" << thread_count << ";" << time << std::endl;
    }
    else {
        time = launch_3pass_proc_none(bdata->ce_start, bdata->ce_stop, block_count, thread_count, bdata->d_input, bdata->d_output, bdata->d_mask, d_pss, cub_pss, chunk_length, chunk_count);
        CUDA_TRY(cudaMemcpy(bdata->h_output, bdata->d_output, bdata->count*sizeof(T), cudaMemcpyDeviceToHost));
        if (!bdata->validate(onecount)) { time = -1; }
        result_data << (bdata->count * sizeof(T)) << ";" << bdata->p << ";" << (cub_pss ? "3pass_fproc_none" : "3pass_pproc_none") << ";" << chunk_length << ";" << block_count << ";" << thread_count << ";" << time << std::endl;
    }
}

// runs all kernels with various parameters at the supplied size
template <typename T>
void run_sized_benchmarks(int cuda_dev_id, std::ofstream& result_data, uint64_t datacount, MaskType maskt, double marg) {
    cudaDeviceProp deviceProp;
    CUDA_TRY(cudaGetDeviceProperties(&deviceProp, cuda_dev_id));
    uint32_t sm_count = deviceProp.multiProcessorCount;

    benchmark_data<T> bdata(datacount); // 1<<18 = 1MiB worth of elems (1<<28 = 2GiB)
    uint32_t onecount = bdata.generate_mask(maskt, marg);
    
    uint32_t max_chunk_count = bdata.count / 32;
    uint32_t* d_pss; // prefix sum scan buffer on device
    CUDA_TRY(cudaMalloc(&d_pss, max_chunk_count*sizeof(uint32_t)));
    uint32_t* d_pss2; // prefix sum scan buffer on device for pss2 tests
    CUDA_TRY(cudaMalloc(&d_pss2, max_chunk_count*sizeof(uint32_t)));
    uint32_t* d_popc; // popc for optimizing 1024bit skips in sparse masks
    CUDA_TRY(cudaMalloc(&d_popc, (bdata.count/1024)*sizeof(uint32_t)));
    uint32_t* d_pss_total; // pss total
    CUDA_TRY(cudaMalloc(&d_pss_total, sizeof(uint32_t)));
    CUDA_TRY(cudaMemset(d_pss_total, 0x00, sizeof(uint32_t)));

    launch_3pass_popc_none(bdata.ce_start, bdata.ce_stop, 0, 256, bdata.d_mask, d_popc, 1024, bdata.count/1024); // popc for 1024bit chunks as skips

    //BENCHMARK 3pass variants and cub pss step
    std::array<uint32_t, 8> block_counts{0, sm_count, 2 * sm_count, 4 * sm_count, 8 * sm_count, 16 * sm_count, 32 * sm_count, 64 * sm_count};
    std::array<uint32_t, 6> thread_counts{32, 64, 128, 256, 512, 1024};
    std::array<uint32_t, 6> chunk_lengths{32, 64, 128, 256, 512, 1024};
    for(int i = 0; i < RUNS_MEASURE; i++){
        for (auto chunk_length : chunk_lengths) {
            uint32_t chunk_count = bdata.count / chunk_length;
            for (auto block_count : block_counts) {
                for (auto thread_count : thread_counts) {
                    run_single_benchmark<T>(&bdata, result_data, onecount, d_pss, d_pss2, d_pss_total, d_popc, false, false, chunk_length, chunk_count, block_count, thread_count);
                    run_single_benchmark<T>(&bdata, result_data, onecount, d_pss, d_pss2, d_pss_total, d_popc, false, true, chunk_length, chunk_count, block_count, thread_count);
                    run_single_benchmark<T>(&bdata, result_data, onecount, d_pss, d_pss2, d_pss_total, d_popc, true, false, chunk_length, chunk_count, block_count, thread_count);
                    run_single_benchmark<T>(&bdata, result_data, onecount, d_pss, d_pss2, d_pss_total, d_popc, true, true, chunk_length, chunk_count, block_count, thread_count);
                }
            }
        }
    }

    float time;
    //BENCHMARK cub_flagged_bytemask
    for (int r = 0; r < RUNS_MEASURE; r++) {
        CUDA_TRY(cudaMemset(bdata.d_output, 0x00, bdata.count)); // reset output between runs
        time = launch_cub_flagged_bytemask(bdata.ce_start, bdata.ce_stop, bdata.d_input, bdata.d_output, bdata.h_mask, d_pss_total, bdata.count);
        CUDA_TRY(cudaMemcpy(bdata.h_output, bdata.d_output, bdata.count*sizeof(T), cudaMemcpyDeviceToHost));
        if (!bdata.validate(onecount)) { time = -1; }
        result_data << (bdata.count*sizeof(T)) << ";" << bdata.p << ";cub_flagged_bytemask;0;0;0;" << time << "\n";
    }

    //BENCHMARK cub_flagged_biterator
    for (int r = 0; r < RUNS_MEASURE; r++) {
        CUDA_TRY(cudaMemset(bdata.d_output, 0x00, bdata.count)); // reset output between runs
        time = launch_cub_flagged_biterator(bdata.ce_start, bdata.ce_stop, bdata.d_input, bdata.d_output, bdata.d_mask, d_pss_total, bdata.count);
        CUDA_TRY(cudaMemcpy(bdata.h_output, bdata.d_output, bdata.count*sizeof(T), cudaMemcpyDeviceToHost));
        if (!bdata.validate(onecount)) { time = -1; }
        result_data << (bdata.count*sizeof(T)) << ";" << bdata.p << ";cub_flagged_biterator;0;0;0;" << time << "\n";
    }

    //BENCHMARK single_thread
    for (int r = 0; r < 2; r++) {
        CUDA_TRY(cudaMemset(bdata.d_output, 0x00, bdata.count)); // reset output between runs
        time = launch_singlethread(bdata.ce_start, bdata.ce_stop, bdata.d_input, bdata.d_mask, bdata.d_output, bdata.count);
        CUDA_TRY(cudaMemcpy(bdata.h_output, bdata.d_output, bdata.count*sizeof(T), cudaMemcpyDeviceToHost));
        if (!bdata.validate(onecount)) { time = -1; }
        result_data << (bdata.count*sizeof(T)) << ";" << bdata.p << ";single_thread;0;1;1;" << time << "\n";
    }

    // free temporary device resources
    CUDA_TRY(cudaFree(d_pss));
    CUDA_TRY(cudaFree(d_pss_total));

#ifdef  AVXPOWER
    return;
    //BENCHMARK avx cpu
    for (int r = 0; r < RUNS_MEASURE; r++) {
        time = launch_avx_compressstore(bdata.h_input, bdata.h_mask, bdata.h_output, bdata.count);
        if (!bdata.validate(onecount)) { time = -1; }
        result_data << (bdata.count*sizeof(T)) << ";" << bdata.p << ";avx512;0;0;0;" << time << "\n"; // should be chunklength 512 but is 0 for graphs
    }
#endif
}

#endif
