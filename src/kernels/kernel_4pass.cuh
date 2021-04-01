#ifndef KERNEL_4PASS_CUH
#define KERNEL_4PASS_CUH

#include <cstdint>

struct iovRow {
    uint32_t chunk_id;
    uint32_t mask_repr;
};

__global__ void kernel_4pass_popc_monolithic(uint8_t* mask, uint32_t* pss, iovRow* iov, uint32_t chunk_length, uint32_t chunk_count)
{
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thread index = chunk id
    if (tid >= chunk_count) {
        return;
    }
    uint32_t idx = (chunk_length/32) * tid; // index for 1st 32bit-element of this chunk
    // assuming chunk_length to be multiple of 32
    uint32_t popcount = 0;
    for (int i = 0; i < chunk_length/32; i++) {
        popcount += __popc(reinterpret_cast<uint32_t*>(mask)[idx+i]);
    }
    pss[tid] = popcount;
    iov[tid] = iovRow{tid, popcount};
}

__global__ void kernel_4pass_popc_striding(uint8_t* mask, uint32_t* pss, iovRow* iov, uint32_t chunk_length, uint32_t chunk_count)
{
    for (uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        uint32_t idx = (chunk_length/32) * tid; // index for 1st 32bit-element of this chunk
        // assuming chunk_length to be multiple of 32
        uint32_t popcount = 0;
        for (int i = 0; i < chunk_length/32; i++) {
            popcount += __popc(reinterpret_cast<uint32_t*>(mask)[idx+i]);
        }
        pss[tid] = popcount;
        iov[tid] = iovRow{tid, popcount};
    }
}

void launch_4pass_popc(uint32_t blockcount, uint32_t threadcount, uint8_t* d_mask, uint32_t* d_pss, iovRow* d_iov, uint32_t chunk_length, uint32_t chunk_count)
{
    if (blockcount == 0) {
        blockcount = (chunk_count/threadcount)+1;
        kernel_4pass_popc_monolithic<<<blockcount, threadcount>>>(d_mask, d_pss, d_iov, chunk_length, chunk_count);
    } else {
        kernel_4pass_popc_striding<<<blockcount, threadcount>>>(d_mask, d_pss, d_iov, chunk_length, chunk_count);
    }
}

//TODO template by blockdim for static shared memory allocation
//TODO use shared memory reduction
__global__ void kernel_4pass_pss_monolithic(uint32_t* pss, uint8_t depth, uint32_t chunk_count, uint32_t* out_count)
{
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    tid *= (1<<depth);
}

__global__ void kernel_4pass_pss_striding(uint32_t* pss, uint8_t depth, uint32_t chunk_count, uint32_t* out_count)
{
    
}

void launch_4pass_pss(uint32_t blockcount, uint32_t threadcount, uint32_t* d_pss, uint32_t chunk_count, uint32_t* d_out_count)
{
    //TODO repeat for reduction depths
    if (blockcount == 0) {
        blockcount = (chunk_count/threadcount)+1;
        kernel_4pass_pss_monolithic<<<blockcount, threadcount>>>(d_pss, 0, chunk_count, d_out_count);
    } else {
        kernel_4pass_pss_striding<<<blockcount, threadcount>>>(d_pss, 0, chunk_count, d_out_count);
    }
}

template <typename T>
__global__ void kernel_4pass_fproc_monolithic(T* input, T* output, uint8_t* mask, uint32_t* pss, uint32_t chunk_length, uint32_t chunk_count) {
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= chunk_count) {
        return;
    }
    uint32_t out_idx = pss[tid];
    uint32_t element_idx = tid*chunk_length/8;
    for (uint32_t i = element_idx; i < element_idx+chunk_length/8; i++) {
        uint8_t acc = mask[i];
        for (int j = 7; j >= 0; j--) {
            uint64_t idx = i*8 + (7-j);
            bool v = 0b1 & (acc>>j);
            if (v) {
                output[out_idx++] = input[idx];
            }
        }
    }
}

template <typename T>
__global__ void kernel_4pass_fproc_striding(T* input, T* output, uint8_t* mask, uint32_t* pss, uint32_t chunk_length, uint32_t chunk_count) {

}

// full processing (i.e. pss is complete scan)
template <typename T>
void launch_4pass_fproc(uint32_t blockcount, uint32_t threadcount, T* d_input, T* d_output, uint8_t* d_mask, uint32_t* d_pss, uint32_t chunk_length, uint32_t chunk_count)
{
    if (blockcount == 0) {
        blockcount = (chunk_count/threadcount)+1;
        kernel_4pass_fproc_monolithic<<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length, chunk_count);
    } else {
        kernel_4pass_fproc_striding<<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length, chunk_count);
    }
}

#endif
