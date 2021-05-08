#include <bitset>
#include <cub/cub.cuh>
#include <cstdint>
#include <iostream>
#include <stdio.h>

#include "benchmark_data.cuh"
#include "benchmark_experiment.cuh"
#include "cub_wraps.cuh"
#include "cuda_time.cuh"
#include "cuda_try.cuh"
#include "streaming_3pass.cuh"
#include "kernels/kernel_3pass.cuh"
#include "kernels/kernel_copy_add.cuh"
#include "kernels/kernel_singlethread.cuh"

template <typename T>
void gpu_buffer_print(T* d_buffer, uint32_t offset, uint32_t count)
{
    T* h_buffer = static_cast<T*>(malloc(count*sizeof(T)));
    CUDA_TRY(cudaMemcpy(h_buffer, d_buffer+offset, count*sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < count; i++) {
        std::bitset<sizeof(T)*8> bits(h_buffer[i]);
        std::cout << bits << " - " << unsigned(h_buffer[i]) << "\n";
    }
    free(h_buffer);
}

void run_sandbox()
{
    benchmark_data<uint64_t> bdata(1<<30);
    uint32_t onecount = bdata.generate_mask(MASKTYPE_UNIFORM, 0.5);
    uint32_t chunk_length = 1024;
    uint32_t chunk_count = bdata.count / chunk_length;
    uint32_t max_chunk_count = bdata.count / 32;
    uint32_t* d_pss;
    CUDA_TRY(cudaMalloc(&d_pss, max_chunk_count*sizeof(uint32_t)));
    uint32_t* d_popc;
    CUDA_TRY(cudaMalloc(&d_popc, max_chunk_count*sizeof(uint32_t)));

    uint32_t* d_pss_total;
    CUDA_TRY(cudaMalloc(&d_pss_total, sizeof(uint32_t)));
    CUDA_TRY(cudaMemset(d_pss_total, 0x00, sizeof(uint32_t)));

    std::cout << "setup\n";


    float time = 0;
    int c = 10;
    for (int i = 0; i < c; i++) {

        // time += launch_3pass_popc_none(bdata.ce_start, bdata.ce_stop, 0, 256, bdata.d_mask, d_pss, chunk_length, chunk_count);
        // time += launch_3pass_pss_gmem(bdata.ce_start, bdata.ce_stop, 0, 512, d_pss, chunk_count, d_pss_total);
        // time += launch_3pass_popc_none(bdata.ce_start, bdata.ce_stop, 0, 512, bdata.d_mask, d_pss, chunk_length, chunk_count);
        // time += launch_3pass_pss_gmem(bdata.ce_start, bdata.ce_stop, 0, 1024, d_pss, chunk_count, d_pss_total);
        // time += launch_3pass_popc_none(bdata.ce_start, bdata.ce_stop, 0, 1024, bdata.d_mask, d_pss, chunk_length, chunk_count);
        // time += launch_cub_pss(0, bdata.ce_start, bdata.ce_stop, d_pss, d_pss_total, chunk_count);
        //time += launch_3pass_popc_none(bdata.ce_start, bdata.ce_stop, 0, 1024, bdata.d_mask, d_popc, 1024, bdata.count/1024);
        //time += launch_3pass_proc_true(bdata.ce_start, bdata.ce_stop, 0, 1024, bdata.d_input, bdata.d_output, bdata.d_mask, d_pss, true, d_popc, chunk_length, chunk_count);
        // time += launch_async_streaming_3pass(bdata.d_input, bdata.d_mask, bdata.d_output, bdata.count, 8);
        // CUDA_TRY(cudaMemcpy(bdata.h_output, bdata.d_output, sizeof(uint64_t)*bdata.count, cudaMemcpyDeviceToHost));
        // bdata.validate(bdata.count);

    }
    std::cout << "time: " << time/c << "\n";

    CUDA_TRY(cudaFree(d_pss_total));
    CUDA_TRY(cudaFree(d_popc));
    CUDA_TRY(cudaFree(d_pss));

    printf("done\n");
    exit(0);

    float min_time = -1;
    std::array<uint32_t, 6> thread_counts{32, 64, 128, 256, 512, 1024};
    std::array<uint32_t, 5> block_counts{0, 256, 1024, 2048, 4096};
    for (auto block_count : block_counts) {
        for (auto thread_count : thread_counts) {
            time = 0;
            for (int i = 0; i < 20; i++) {
                time += launch_copy_add(bdata.ce_start, bdata.ce_stop, block_count, thread_count, bdata.d_input, bdata.d_output, bdata.d_mask, bdata.count, true);
            }
            time /= 20;
            if (min_time < 0 || min_time > time) {
                min_time = time;
            }
            std::cout << "B#" << block_count << " T#" << thread_count << " : " << time << "\n";
        }
    }
    std::cout << "min_time: " << min_time << "\n";

    printf("done\n");
    exit(0);
}

int main()
{
    int cuda_dev_id = 0;
    CUDA_TRY(cudaSetDevice(cuda_dev_id));

    //run_sandbox();

    std::ofstream result_data;
    result_data.open("result_data.csv");
    if (!result_data) {
        std::cerr << "error: result file could not be opened\n";
        exit(1);
    }
    result_data << "databytes;p;algo;chunklength;blocks;threads;time\n";
    // run from 16MiB to 8GiB in powers of 2
    std::array<double, 7> p_values{0.995, 0.95, 0.5, 0.05, 0.005, 0.0005, 0.00005};
    for (uint64_t datasize = 1<<21; datasize <= 1<<30; datasize <<=1) {
        for (auto p_value : p_values) {
            run_sized_benchmarks<uint64_t>(cuda_dev_id, result_data, datasize, MASKTYPE_UNIFORM, p_value, true);
        }
    }

    // run extra bench excluding slow algos on 8GiB with other masks
    //run_sized_benchmarks<uint64_t>(cuda_dev_id, result_data, 1<<30, MASKTYPE_ZIPF, 1.2, false);
    //run_sized_benchmarks<uint64_t>(cuda_dev_id, result_data, 1<<30, MASKTYPE_BURST, 0.0001, false); // has streaks of ~107k same bits
    
    result_data.close();

    printf("done\n");
    return 0;
}
