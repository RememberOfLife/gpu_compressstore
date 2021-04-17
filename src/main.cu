#include <bitset>
#include <cub/cub.cuh>
#include <cstdint>
#include <iostream>
#include <stdio.h>

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
    CUDA_TRY(cudaSetDevice(0));
    benchmark_data<uint64_t> bdata(1<<29); // 1<<18 = 1MiB worth of elems (1<<28 = 2GiB)
    std::cout << "onecount: " << bdata.generate_mask(MASKTYPE_UNIFORM, 0.5) << "\n";

    run_benchmark_experiment(&bdata);
    std::cout << "done\n";
    return 0;

    // 3 pass algo
    uint32_t chunk_length = 128;
    uint32_t pass1_blockcount = 0;
    uint32_t pass1_threadcount = 256;
    uint32_t pass2_blockcount = 0;
    uint32_t pass2_threadcount = 256;
    uint32_t pass3_blockcount = 0;
    uint32_t pass3_threadcount = 256;

    uint32_t chunk_count = bdata.count / chunk_length;
    uint32_t* d_pss; // prefix sum scan buffer on device
    CUDA_TRY(cudaMalloc(&d_pss, chunk_count*sizeof(uint32_t)));
    iovRow* d_iov; // intermediate optimization vector
    CUDA_TRY(cudaMalloc(&d_iov, chunk_count*sizeof(iovRow)));
    uint32_t* d_pss_total; // pss total
    CUDA_TRY(cudaMalloc(&d_pss_total, sizeof(uint32_t)));
    CUDA_TRY(cudaMemset(d_pss_total, 0x00, sizeof(uint32_t)));
    uint32_t h_pss_total = 0;
    // #1: pop count per chunk and populate IOV
    launch_3pass_popc_iov(bdata.ce_start, bdata.ce_stop, pass1_blockcount, pass1_threadcount, bdata.d_mask, d_pss, d_iov, chunk_length, chunk_count);
    // #2: prefix sum scan (for partial trees)
    launch_3pass_pss_gmem(bdata.ce_start, bdata.ce_stop, pass2_blockcount, pass2_threadcount, d_pss, chunk_count, d_pss_total);
    //launch_cub_pss(bdata.ce_start, bdata.ce_stop, d_pss, d_pss_total, chunk_count); // cub launch as alternative
    CUDA_TRY(cudaMemcpy(&h_pss_total, d_pss_total, sizeof(uint32_t), cudaMemcpyDeviceToHost)); // copy total popcount to host
    double mask_dp = static_cast<double>(h_pss_total) / static_cast<double>(bdata.count); // distribution parameter (assuming uniform distribution)
    std::cout << "MDP: " << mask_dp << "\n";
    // #4: processing of chunks
    int c = 20;
    float t;
    for (int i = 0; i < 3; i++) {
        launch_3pass_proc_iov(bdata.ce_start, bdata.ce_stop, pass3_blockcount, pass3_threadcount, bdata.d_input, bdata.d_output, bdata.d_mask, d_pss, false, d_iov, chunk_length, chunk_count);
    }
    t = 0;
    for (int i = 0; i < c; i++) {
        t += launch_3pass_proc_iov(bdata.ce_start, bdata.ce_stop, pass3_blockcount, pass3_threadcount, bdata.d_input, bdata.d_output, bdata.d_mask, d_pss, false, d_iov, chunk_length, chunk_count);
    }
    std::cout << "timing iov: " << t/c << "\n";

    // free temporary device resources
    CUDA_TRY(cudaFree(d_iov));
    CUDA_TRY(cudaFree(d_pss));
    CUDA_TRY(cudaFree(d_pss_total));


    CUDA_TRY(cudaMemcpy(bdata.h_output, bdata.d_output, bdata.count*sizeof(uint64_t), cudaMemcpyDeviceToHost));
    /*/ print for testing (first 64 elems of input, validation and mask)
    std::cout << "maskset:\n";
    for (int k = 0; k < 1; k++) {
        for (int i = 0; i < 4; i++) {
            std::bitset<8> maskset(bdata.h_mask[k*4+i]);
            std::cout << maskset;
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    for (int i = 0; i < 1024*3; i ++) {
        // mask value for this input
        uint32_t offset32 = i % 8;
        uint32_t base32 = (i-offset32) / 8;
        uint32_t mask32 = reinterpret_cast<uint8_t*>(bdata.h_mask)[base32];
        uint32_t mask = 0b1 & (mask32>>(7-offset32));
        //std::cout << "[" << i << "] " << mask;
        // print number residing there
        uint64_t num = bdata.h_input[i];
        bool fits = (bdata.h_validation[i] == bdata.h_output[i]);
        //std::cout << " - " << num << " - " << bdata.h_validation[i] << " - " << bdata.h_output[i] << " - " << fits << "\n";
    }//*/

    std::cout << "selected: " << h_pss_total << "\n";
    bdata.validate(bdata.count);

    printf("done");
    return 0;
}
