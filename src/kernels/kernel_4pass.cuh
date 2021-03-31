#ifndef KERNEL_4PASS_CUH
#define KERNEL_4PASS_CUH

#include <cstdint>

struct iovRow {
    uint32_t chunk_id;
    uint32_t mask_repr;
};

__global__ void kernel_4pass_popc_monolithic(uint8_t* mask, uint16_t* pss, iovRow* iov, uint32_t chunk_length, uint32_t chunk_count) {
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

__global__ void kernel_4pass_popc_striding(uint8_t* mask, uint16_t* pss, iovRow* iov, uint32_t chunk_length, uint32_t chunk_count) {
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

void launch_4pass_popc(uint32_t blockcount, uint32_t threadcount, uint8_t* d_mask, uint16_t* d_pss, iovRow* d_iov, uint32_t chunk_length, uint32_t chunk_count) {
    if (blockcount == 0) {
        blockcount = (chunk_count/(chunk_length*threadcount))+1;
        kernel_4pass_popc_monolithic<<<blockcount, threadcount>>>(d_mask, d_pss, d_iov, chunk_length, chunk_count);
    } else {
        kernel_4pass_popc_striding<<<blockcount, threadcount>>>(d_mask, d_pss, d_iov, chunk_length, chunk_count);
    }
}

//TODO template by blockdim for static shared memory allocation
//TODO use shared memory reduction
__global__ void kernel_4pass_pss_monolithic(uint16_t* pss, uint8_t depth, uint32_t chunk_count, uint32_t* out_count) {
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    tid *= (1<<depth);
}

__global__ void kernel_4pass_pss_striding(uint16_t* pss, uint8_t depth, uint32_t chunk_count, uint32_t* out_count) {
    
}

void launch_4pass_pss(uint32_t blockcount, uint32_t threadcount, uint16_t* d_pss, uint32_t chunk_count, uint32_t* d_out_count) {
    //TODO repeat for reduction depths
    if (blockcount == 0) {
        blockcount = (chunk_count/threadcount)+1;
        kernel_4pass_pss_monolithic<<<blockcount, threadcount>>>(d_pss, 0, chunk_count, d_out_count);
    } else {
        kernel_4pass_pss_striding<<<blockcount, threadcount>>>(d_pss, 0, chunk_count, d_out_count);
    }
}

#endif
