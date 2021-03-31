#ifndef KERNEL_4PASS_CUH
#define KERNEL_4PASS_CUH

#include <cstdint>

struct iovRow {
    uint32_t chunk_id;
    uint32_t mask_repr;
};

__global__ void kernel_4pass_popcount_monolithic(uint8_t* mask, uint16_t* pss, iovRow* iov, uint32_t chunk_length, uint32_t chunk_count) {
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thread index
    uint32_t idx = (chunk_length/32) * tid; // index for 1st 32bit-element of this chunk
    if (tid >= chunk_count) {
        return;
    }
    // assuming chunk_length to be multiple of 32
    uint16_t popcount = 0;
    for (int i = 0; i < chunk_length/32; i++) {
        popcount += __popc(reinterpret_cast<uint32_t*>(mask)[idx+i]);
    }
    pss[tid] = popcount;
    iov[tid] = iovRow{tid, popcount};
}

__global__ void kernel_4pass_popcount_striding(uint8_t* mask, uint16_t* pss, iovRow* iov, uint32_t chunk_length, uint32_t chunk_count) {
    for (uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        uint32_t idx = (chunk_length/32) * tid; // index for 1st 32bit-element of this chunk
        // assuming chunk_length to be multiple of 32
        uint16_t popcount = 0;
        for (int i = 0; i < chunk_length/32; i++) {
            popcount += __popc(reinterpret_cast<uint32_t*>(mask)[idx+i]);
        }
        pss[tid] = popcount;
        iov[tid] = iovRow{tid, popcount};
    }
}

#endif
