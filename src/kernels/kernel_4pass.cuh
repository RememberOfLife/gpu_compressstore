#ifndef KERNEL_4PASS_CUH
#define KERNEL_4PASS_CUH

#include <cmath>
#include <cstdint>

#include "cuda_time.cuh"

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

float launch_4pass_popc(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    uint8_t* d_mask,
    uint32_t* d_pss,
    iovRow* d_iov,
    uint32_t chunk_length,
    uint32_t chunk_count)
{
    float time;
    if (blockcount == 0) {
        blockcount = (chunk_count/threadcount)+1;
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_4pass_popc_monolithic<<<blockcount, threadcount>>>(d_mask, d_pss, d_iov, chunk_length, chunk_count))
        );
    } else {
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_4pass_popc_striding<<<blockcount, threadcount>>>(d_mask, d_pss, d_iov, chunk_length, chunk_count))
        );
    }
    return time;
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
    uint64_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint64_t stride = (1<<depth);
    tid = 2*tid*stride+stride-1;
    // tid is element id

    // thread loads element at tid and tid+stride
    if (tid >= chunk_count) {
        return;
    }
    uint32_t left_e = pss[tid];
    if (tid+stride < chunk_count) {
        pss[tid+stride] += left_e;
    } else {
        (*out_count) += left_e;
    }
}

__global__ void kernel_4pass_pss_striding(uint32_t* pss, uint8_t depth, uint32_t chunk_count, uint32_t* out_count)
{
    uint64_t stride = (1<<depth);
    for (uint64_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        uint64_t cid = 2*tid*stride+stride-1; // calc chunk id
        if (cid >= chunk_count) {
            return;
        }
        uint32_t left_e = pss[cid];
        if (cid+stride < chunk_count) {
            pss[cid+stride] += left_e;
        } else {
            (*out_count) = left_e;
        }
    }
}

float launch_4pass_pss(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    uint32_t* d_pss,
    uint32_t chunk_count,
    uint32_t* d_out_count)
{
    float time = 0;
    float ptime;
    uint32_t max_depth = 0;
    for (uint32_t chunk_count_p2 = 1; chunk_count_p2 < chunk_count; max_depth++) {
        chunk_count_p2 *= 2;
    }
    if (blockcount == 0) {
        // reduce blockcount every depth iteration
        for (int i = 0; i < max_depth; i++) {
            blockcount = ((chunk_count>>i)/(threadcount*2))+1;
            CUDA_TIME(ce_start, ce_stop, 0, &ptime,
                (kernel_4pass_pss_monolithic<<<blockcount, threadcount>>>(d_pss, i, chunk_count, d_out_count))
            );
            time += ptime;
        }
        // last pass forces result into d_out_count
        CUDA_TIME(ce_start, ce_stop, 0, &ptime, 
            (kernel_4pass_pss_monolithic<<<1, 1>>>(d_pss, static_cast<uint8_t>(max_depth), chunk_count, d_out_count))
        );
        time += ptime;
    } else {
        for (int i = 0; i < max_depth; i++) {
            uint32_t req_blockcount = ((chunk_count>>i)/(threadcount*2))+1;
            if (blockcount > req_blockcount) {
                blockcount = req_blockcount;
            }
            CUDA_TIME(ce_start, ce_stop, 0, &ptime,
                (kernel_4pass_pss_striding<<<blockcount, threadcount>>>(d_pss, i, chunk_count, d_out_count))
            );
            time += ptime;
        }
        // last pass forces result into d_out_count
        CUDA_TIME(ce_start, ce_stop, 0, &ptime,
            (kernel_4pass_pss_monolithic<<<1, 1>>>(d_pss, static_cast<uint8_t>(max_depth), chunk_count, d_out_count))
        );
        time += ptime;
    }
    return time;
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
float launch_4pass_fproc(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    T* d_input,
    T* d_output,
    uint8_t* d_mask,
    uint32_t* d_pss,
    uint32_t chunk_length,
    uint32_t chunk_count)
{
    float time;
    if (blockcount == 0) {
        blockcount = (chunk_count/threadcount)+1;
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_4pass_proc_monolithic<T, true><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length, chunk_count, 0))
        );
    } else {
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_4pass_proc_striding<T, true><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length, chunk_count, 0))
        );
    }
    return time;
}

// processing (for partial pss)
template <typename T>
float launch_4pass_pproc(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    T* d_input,
    T* d_output,
    uint8_t* d_mask,
    uint32_t* d_pss,
    uint32_t chunk_length,
    uint32_t chunk_count)
{
    float time;
    uint32_t chunk_count_p2 = 1;
    while (chunk_count_p2 < chunk_count) {
        chunk_count_p2 *= 2;
    }
    if (blockcount == 0) {
        blockcount = (chunk_count/threadcount)+1;
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_4pass_proc_monolithic<T, false><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length, chunk_count, chunk_count_p2))
        );
    } else {
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_4pass_proc_striding<T, false><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length, chunk_count, chunk_count_p2))
        );
    }
    return time;
}

#endif
