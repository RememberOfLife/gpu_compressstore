#include <bitset>
#include <cstdint>
#include <iostream>
#include <stdio.h>

#include "benchmark_data.cuh"
#include "cuda_try.cuh"
#include "kernels.cuh"

int main()
{
    benchmark_data bdata(true, 64); // 2<<17 = 1MiB worth of elems (2<<27 = 2GiB)
    bdata.generate_mask(MASKTYPE_UNIFORM, 0.5);

    kernel_singlethread<<<1,1>>>(bdata.d_input, bdata.d_mask, bdata.d_output, bdata.size);
    CUDA_TRY(cudaMemcpy(bdata.h_output, bdata.d_output, bdata.size * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    
    // print for testing (first 64 elems of input, validation and mask)
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

    bdata.validate(bdata.size);

    printf("done");
    return 0;
}
