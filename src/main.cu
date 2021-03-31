#include <bitset>
#include <cub/cub.cuh>
#include <cstdint>
#include <iostream>
#include <stdio.h>

#include "benchmark_data.cuh"
#include "cuda_try.cuh"
#include "kernels/kernel_4pass.cuh"
#include "kernels/kernel_singlethread.cuh"

template <typename T>
void gpu_buffer_print(T* d_buffer, uint32_t offset, uint32_t count) {
    T* h_buffer = static_cast<T*>(malloc(count*sizeof(T)));
    CUDA_TRY(cudaMemcpy(h_buffer, d_buffer, count*sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = offset; i < offset+count; i++) {
        std::bitset<sizeof(T)*8> bits(h_buffer[i]);
        std::cout << bits << " - " << h_buffer[i] << "\n";
    }
    free(h_buffer);
}

int main()
{
    benchmark_data<uint64_t> bdata(true, 128); // 2<<17 = 1MiB worth of elems (2<<27 = 2GiB)
    bdata.generate_mask(MASKTYPE_UNIFORM, 0.5);

    
    // 4 pass algo
    uint32_t chunk_length = 32;
    uint32_t pass1_blockcount = 0;
    uint32_t pass1_threadcount = 256;
    //uint32_t pass2_blockcount = 0;
    //uint32_t pass2_threadcount = 256;

    uint32_t chunk_count = bdata.count / chunk_length;
    uint16_t* d_pss; // prefix sum scan buffer on device
    CUDA_TRY(cudaMalloc(&d_pss, chunk_count*sizeof(uint16_t)));
    iovRow* d_iov; // intermediate optimization vector
    CUDA_TRY(cudaMalloc(&d_iov, chunk_count*sizeof(iovRow)));
    uint32_t* d_pss_total; // pss total
    CUDA_TRY(cudaMalloc(&d_pss_total, sizeof(uint32_t)));
    // #1: pop count per chunk and populate IOV
    launch_4pass_popc(pass1_blockcount, pass1_threadcount, bdata.d_mask, d_pss, d_iov, chunk_length, chunk_count);
    // #2: prefix sum scan (for partial trees)
    //launch_4pass_pss(pass2_blockcount, pass2_threadcount, d_pss, chunk_count, d_pss_total);
    {
        // use cub pss for now
        uint16_t* d_pss_tmp;
        CUDA_TRY(cudaMalloc(&d_pss_tmp, chunk_count*sizeof(uint16_t)));
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        CUDA_TRY(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_pss, d_pss_tmp, chunk_count));
        CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        CUDA_TRY(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_pss, d_pss_tmp, chunk_count));
        CUDA_TRY(cudaFree(d_temp_storage));
        //uint16_t* d_pss_die = d_pss;
        //d_pss = d_pss_tmp;
        CUDA_TRY(cudaMemcpy(d_pss, d_pss_tmp, chunk_count*sizeof(uint16_t), cudaMemcpyDeviceToDevice));
        CUDA_TRY(cudaFree(d_pss_tmp));
    }

    gpu_buffer_print(d_pss, 0, 4);
    gpu_buffer_print(reinterpret_cast<uint32_t*>(d_iov), 0, 4);
    // #3: optimization pass (sort or bucket skip launch)
    // #4: processing of chunks

    // free temporary device resources
    CUDA_TRY(cudaFree(d_iov));
    CUDA_TRY(cudaFree(d_pss));


    CUDA_TRY(cudaMemcpy(bdata.h_output, bdata.d_output, bdata.count*sizeof(uint64_t), cudaMemcpyDeviceToHost));
    /*/ print for testing (first 64 elems of input, validation and mask)
    std::bitset<8> maskset(bdata.h_mask[0]);
    std::cout << "maskset: " << maskset << "\n\n";
    for (int i = 0; i < 64; i ++) {
        // mask value for this input
        uint32_t offset32 = i % 8;
        uint32_t base32 = (i-offset32) / 8;
        uint32_t mask32 = reinterpret_cast<uint8_t*>(bdata.h_mask)[base32];
        uint32_t mask = 0b1 & (mask32>>(7-offset32));
        std::cout << mask;
        // print number residing there
        uint64_t num = bdata.h_input[i];
        std::bitset<64> numbs(num);
        std::bitset<64> valid(bdata.h_validation[i]);
        std::bitset<64> gout(bdata.h_output[i]);
        std::cout << " - " << numbs << " - " << valid << " - " << gout << "\n";
    }//*/

    //bdata.validate(bdata.count);

    printf("done");
    return 0;
}
