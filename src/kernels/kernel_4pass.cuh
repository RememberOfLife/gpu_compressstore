#ifndef KERNEL_4PASS_CUH
#define KERNEL_4PASS_CUH

#include <cmath>
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

__global__ void kernel_4pass_pssskip(uint32_t* pss, uint32_t* pss_total, uint32_t chunk_count)
{
    *pss_total += pss[chunk_count-1];
}
void launch_4pass_pssskip(uint32_t* d_pss, uint32_t* d_pss_total, uint32_t chunk_count)
{
    kernel_4pass_pssskip<<<1,1>>>(d_pss, d_pss_total, chunk_count);
}

//TODO template by blockdim*chunk_count for static shared memory allocation
//TODO use shared memory reduction
__global__ void kernel_4pass_pss_monolithic(uint32_t* pss, uint8_t depth, uint32_t chunk_count, uint32_t* out_count)
{
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint32_t stride = (1<<depth);
    tid = 2*tid*stride+stride-1;
    // tid is element id

    // thread loads element at tid and tid+stride
    uint32_t left_e = 0;
    uint32_t right_e = 0;
    bool dead_chunk = false;
    if (tid < chunk_count) {
        left_e = pss[tid];
    } else {
        return;
    }
    if (tid+stride < chunk_count) {
        right_e = pss[tid+stride];
    } else {
        dead_chunk = true;
    }
    uint32_t total = left_e + right_e + (dead_chunk ? (*out_count) : 0);
    if (tid+stride < chunk_count) {
        pss[tid+stride] = total;
    } else {
        (*out_count) = total;
    }
}

__global__ void kernel_4pass_pss_striding(uint32_t* pss, uint8_t depth, uint32_t chunk_count, uint32_t* out_count)
{
    uint32_t stride = (1<<depth);
    for (uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        tid = 2*tid*stride+stride-1;
        uint32_t left_e = 0;
        uint32_t right_e = 0;
        bool dead_chunk = false;
        if (tid < chunk_count) {
            left_e = pss[tid];
        } else {
            return;
        }
        if (tid+stride < chunk_count) {
            right_e = pss[tid+stride];
        } else {
            dead_chunk = true;
        }
        uint32_t total = left_e + right_e + (dead_chunk ? (*out_count) : 0);
        if (tid+stride < chunk_count) {
            pss[tid+stride] = total;
        } else {
            (*out_count) = total;
        }
    }
}

void launch_4pass_pss(uint32_t blockcount, uint32_t threadcount, uint32_t* d_pss, uint32_t chunk_count, uint32_t* d_out_count)
{
    //TODO repeat for reduction depths
    if (blockcount == 0) {
        blockcount = (chunk_count/(threadcount*2))+1;
        double maxdepth = log2(chunk_count)+1;
        // reduce blockcount every depth iteration
        for (int i = 0; i < maxdepth; i++) {
            kernel_4pass_pss_monolithic<<<blockcount, threadcount>>>(d_pss, i, chunk_count, d_out_count);
        }
        // last pass forces result into d_out_count
        kernel_4pass_pss_monolithic<<<1, 1>>>(d_pss, static_cast<uint8_t>(maxdepth), chunk_count, d_out_count);
    } else {
        kernel_4pass_pss_striding<<<blockcount, threadcount>>>(d_pss, 0, chunk_count, d_out_count);
    }
}

__device__ uint32_t d_4pass_pproc_pssidx(uint32_t thread_idx, uint32_t* pss, uint32_t chunk_count_p2)
{
    chunk_count_p2 /= 2; // start by trying the subtree with length of half the next rounded up power of 2 of chunk_count
    uint32_t consumed = 0; // length of subtrees already fit inside idx_acc
    uint32_t idx_acc = 0; // assumed starting position for this chunk
    while (chunk_count_p2 >= 1) {
        if (thread_idx >= consumed+chunk_count_p2) {
            // partial tree [consumed, consumed+chunk_count_p2] fits into left side of thread_idx
            idx_acc += pss[consumed+chunk_count_p2-1];
            consumed += chunk_count_p2;
        }
        chunk_count_p2 /= 2;
    }
    return idx_acc;
}

template <typename T, bool complete_pss>
__global__ void kernel_4pass_proc_monolithic(T* input, T* output, uint8_t* mask, uint32_t* pss, uint32_t chunk_length, uint32_t chunk_count, uint32_t chunk_count_p2)
{
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= chunk_count) {
        return;
    }
    uint32_t out_idx;
    if (complete_pss) {
        out_idx = pss[tid];
    } else {
        out_idx = d_4pass_pproc_pssidx(tid, pss, chunk_count_p2);
    }
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

template <typename T, bool complete_pss>
__global__ void kernel_4pass_proc_striding(T* input, T* output, uint8_t* mask, uint32_t* pss, uint32_t chunk_length, uint32_t chunk_count, uint32_t chunk_count_p2)
{
    for (uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        uint32_t out_idx;
        if (complete_pss) {
            out_idx = pss[tid];
        } else {
            out_idx = d_4pass_pproc_pssidx(tid, pss, chunk_count_p2);
        }
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
}

// processing (for complete pss)
template <typename T>
void launch_4pass_fproc(uint32_t blockcount, uint32_t threadcount, T* d_input, T* d_output, uint8_t* d_mask, uint32_t* d_pss, uint32_t chunk_length, uint32_t chunk_count)
{
    if (blockcount == 0) {
        blockcount = (chunk_count/threadcount)+1;
        kernel_4pass_proc_monolithic<T, true><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length, chunk_count, 0);
    } else {
        kernel_4pass_proc_striding<T, true><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length, chunk_count, 0);
    }
}

// processing (for partial pss)
template <typename T>
void launch_4pass_pproc(uint32_t blockcount, uint32_t threadcount, T* d_input, T* d_output, uint8_t* d_mask, uint32_t* d_pss, uint32_t chunk_length, uint32_t chunk_count)
{
    uint32_t chunk_count_ps = 1;
    while (chunk_count_ps < chunk_count) {
        chunk_count_ps *= 2;
    }
    if (blockcount == 0) {
        blockcount = (chunk_count/threadcount)+1;
        kernel_4pass_proc_monolithic<T, false><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length, chunk_count, chunk_count_ps);
    } else {
        kernel_4pass_proc_striding<T, false><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length, chunk_count, chunk_count_ps);
    }
}

#endif
