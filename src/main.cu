#include <bitset>
#include <cub/cub.cuh>
#include <cstdint>
#include <iostream>
#include <stdio.h>

#include "avx_wrap.cuh"
#include "benchmark_data.cuh"
#include "benchmark_experiment.cuh"
#include "cub_wraps.cuh"
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

int main()
{
    int cuda_dev_id = 0;
    CUDA_TRY(cudaSetDevice(cuda_dev_id));

    // sandbox
    benchmark_data<uint64_t> bdata(1<<28);
    bdata.generate_mask(MASKTYPE_UNIFORM, 0.5);

#ifdef AVXPOWER

    for (int i = 0; i < 20; i++) {
        std::cout << "time: " << launch_avx_compressstore(bdata.h_input, bdata.h_mask, bdata.h_output, bdata.count) << "\n";
        bdata.validate(bdata.count);
    }

#endif

    printf("done\n");
    return 0;


    std::ofstream result_data;
    result_data.open("result_data.csv");
    if (!result_data) {
        std::cerr << "error: result file could not be opened\n";
        exit(1);
    }
    result_data << "datasize;algo;chunklength;blocks;threads;time\n";

    for (uint32_t datasize = 1<<21; datasize <= 1<<29; datasize <<=1) {
        run_sized_benchmarks<uint64_t>(cuda_dev_id, result_data, datasize, MASKTYPE_UNIFORM, 0.5);
    }
    
    result_data.close();

    printf("done");
    return 0;
}
