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
#include "kernels/kernel_3pass.cuh"
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

void run_sandbox() {
    benchmark_data<uint64_t> bdata(1<<29);
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

    float time;
    int c = 20;
    for (int i = 0; i < c; i++) {

        time += launch_3pass_popc_none(bdata.ce_start, bdata.ce_stop, 0, 256, bdata.d_mask, d_pss, chunk_length, chunk_count);
        time += launch_3pass_popc_none(bdata.ce_start, bdata.ce_stop, 0, 256, bdata.d_mask, d_popc, 1024, bdata.count/1024);
        time += launch_cub_pss(bdata.ce_start, bdata.ce_stop, d_pss, d_pss_total, chunk_count);
        time += launch_3pass_proc_true(bdata.ce_start, bdata.ce_stop, 0, 256, bdata.d_input, bdata.d_output, bdata.d_mask, d_pss, true, d_popc, chunk_length, chunk_count);
        CUDA_TRY(cudaMemcpy(bdata.h_output, bdata.d_output, sizeof(uint64_t)*bdata.count, cudaMemcpyDeviceToHost));
        bdata.validate(bdata.count);

    }
    std::cout << "time: " << time/c << "\n";

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
    result_data << "datasize;p;algo;chunklength;blocks;threads;time\n";
    // run from 16MiB to  in powers of 8GiB
    //for (uint32_t datasize = 1<<21; datasize <= 1<<30; datasize <<=1) {
    {uint32_t datasize = 1<<21;
        run_sized_benchmarks<uint64_t>(cuda_dev_id, result_data, datasize, MASKTYPE_UNIFORM, 0.5);
        run_sized_benchmarks<uint64_t>(cuda_dev_id, result_data, datasize, MASKTYPE_UNIFORM, 0.05);
        run_sized_benchmarks<uint64_t>(cuda_dev_id, result_data, datasize, MASKTYPE_UNIFORM, 0.005);
        run_sized_benchmarks<uint64_t>(cuda_dev_id, result_data, datasize, MASKTYPE_UNIFORM, 0.0005);
    }
    
    result_data.close();

    printf("done");
    return 0;
}
