#ifndef KERNEL_4PASS_CUH
#define KERNEL_4PASS_CUH

#include <cmath>
#include <cstdint>

#include "cuda_time.cuh"

struct iovRow {
    uint32_t chunk_id;
    uint32_t mask_repr;
};

template <bool write_iov>
__global__ void kernel_4pass_popc_simple_monolithic(uint8_t* mask, uint32_t* pss, iovRow* iov, uint32_t chunk_length32, uint32_t chunk_count)
{
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thread index = chunk id
    if (tid >= chunk_count) {
        return;
    }
    uint32_t idx = chunk_length32 * tid; // index for 1st 32bit-element of this chunk
    // assuming chunk_length to be multiple of 32
    uint32_t popcount = 0;
    for (int i = 0; i < chunk_length32; i++) {
        popcount += __popc(reinterpret_cast<uint32_t*>(mask)[idx+i]);
    }
    pss[tid] = popcount;
    if (write_iov) {
        iov[tid] = iovRow{tid, popcount};
    }
}

template <bool write_iov>
__global__ void kernel_4pass_popc_simple_striding(uint8_t* mask, uint32_t* pss, iovRow* iov, uint32_t chunk_length32, uint32_t chunk_count)
{
    for (uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        uint32_t idx = chunk_length32 * tid; // index for 1st 32bit-element of this chunk
        // assuming chunk_length to be multiple of 32
        uint32_t popcount = 0;
        for (int i = 0; i < chunk_length32; i++) {
            popcount += __popc(reinterpret_cast<uint32_t*>(mask)[idx+i]);
        }
        pss[tid] = popcount;
        if (write_iov) {
            iov[tid] = iovRow{tid, popcount};
        }
    }
}

float launch_4pass_popc_none(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    uint8_t* d_mask,
    uint32_t* d_pss,
    iovRow* d_iov, // unused in none variant
    uint32_t chunk_length,
    uint32_t chunk_count)
{
    float time;
    uint32_t chunk_length32 = chunk_length / 32;
    if (blockcount == 0) {
        blockcount = (chunk_count/threadcount)+1;
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_4pass_popc_simple_monolithic<false><<<blockcount, threadcount>>>(d_mask, d_pss, d_iov, chunk_length32, chunk_count))
        );
    } else {
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_4pass_popc_simple_striding<false><<<blockcount, threadcount>>>(d_mask, d_pss, d_iov, chunk_length32, chunk_count))
        );
    }
    return time;
}

float launch_4pass_popc_simple(
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
    uint32_t chunk_length32 = chunk_length / 32;
    if (blockcount == 0) {
        blockcount = (chunk_count/threadcount)+1;
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_4pass_popc_simple_monolithic<true><<<blockcount, threadcount>>>(d_mask, d_pss, d_iov, chunk_length32, chunk_count))
        );
    } else {
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_4pass_popc_simple_striding<true><<<blockcount, threadcount>>>(d_mask, d_pss, d_iov, chunk_length32, chunk_count))
        );
    }
    return time;
}

#define KERNEL_4PASS_POPC_IOV_PROCBYTE(byte)                                                                      \
    mask_byte = static_cast<uint8_t>(byte);                                                                       \
    popcount += __popc(mask_byte);                                                                                \
    byte_acc_ones &= mask_byte;                                                                                   \
    byte_acc_zero |= mask_byte;                                                                                   \
    byte_acc_count--;                                                                                             \
    if (byte_acc_count == 0) {                                                                                    \
        mask_repr |= ( ((byte_acc_ones == 0xFF)&0b1)<<1 | ((byte_acc_zero == 0x00)&0b1) )<<(mask_repr_pos*2);     \
        mask_repr_pos--;                                                                                          \
        byte_acc_count = mask_repr_power>>((mask_repr_pos*2 + 1) < mask_repr_switch);                             \
        byte_acc_ones = 0xFF;                                                                                     \
        byte_acc_zero = 0x00;                                                                                     \
    }

__global__ void kernel_4pass_popc_iov_monolithic(
    uint8_t* mask,
    uint32_t* pss,
    iovRow* iov,
    uint32_t chunk_length32,
    uint32_t chunk_count,
    uint8_t mask_repr_power,
    uint8_t mask_repr_switch)
{
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thread index = chunk id
    if (tid >= chunk_count) {
        return;
    }
    uint32_t idx = chunk_length32 * tid; // index for 1st 32bit-element of this chunk
    // assuming chunk_length to be multiple of 32
    uint32_t popcount = 0;
    uint32_t mask_repr = 0;
    // splicing together these vars in one uint32 gains no performance benefit, same for arguments
    uint8_t mask_repr_pos = 15; // position of 2-bit tuple
    uint8_t byte_acc_count = mask_repr_power;
    uint8_t byte_acc_ones = 0xFF;
    uint8_t byte_acc_zero = 0x00;
    for (int i = 0; i < chunk_length32; i++) {
        uchar4 mask_elem = reinterpret_cast<uchar4*>(mask)[idx+i];
        uint8_t mask_byte;
        KERNEL_4PASS_POPC_IOV_PROCBYTE(mask_elem.x);
        KERNEL_4PASS_POPC_IOV_PROCBYTE(mask_elem.y);
        KERNEL_4PASS_POPC_IOV_PROCBYTE(mask_elem.z);
        KERNEL_4PASS_POPC_IOV_PROCBYTE(mask_elem.w);
    }
    pss[tid] = popcount;
    iov[tid] = iovRow{tid, mask_repr};
}

__global__ void kernel_4pass_popc_iov_striding(
    uint8_t* mask,
    uint32_t* pss,
    iovRow* iov,
    uint32_t chunk_length32,
    uint32_t chunk_count,
    uint8_t mask_repr_power,
    uint8_t mask_repr_switch)
{
    for (uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        uint32_t idx = chunk_length32 * tid; // index for 1st 32bit-element of this chunk
        // assuming chunk_length to be multiple of 32
        uint32_t popcount = 0;
        uint32_t mask_repr = 0;
        // splicing together these vars in one uint32 gains no performance benefit, same for arguments
        uint8_t mask_repr_pos = 15; // position of 2-bit tuple
        uint8_t byte_acc_count = mask_repr_power;
        uint8_t byte_acc_ones = 0xFF;
        uint8_t byte_acc_zero = 0x00;
        for (int i = 0; i < chunk_length32; i++) {
            uchar4 mask_elem = reinterpret_cast<uchar4*>(mask)[idx+i];
            uint8_t mask_byte;
            KERNEL_4PASS_POPC_IOV_PROCBYTE(mask_elem.x);
            KERNEL_4PASS_POPC_IOV_PROCBYTE(mask_elem.y);
            KERNEL_4PASS_POPC_IOV_PROCBYTE(mask_elem.z);
            KERNEL_4PASS_POPC_IOV_PROCBYTE(mask_elem.w);
        }
        pss[tid] = popcount;
        iov[tid] = iovRow{tid, mask_repr};
    }
}

float launch_4pass_popc_iov(
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
    uint32_t chunk_length32 = chunk_length / 32;
    uint8_t mask_repr_bytes = chunk_length / 8; // number of bytes represented in the mask_repr
    uint8_t mask_repr_power = 1; // bytes represented by every element before/at the switch (half this after the switch)
    while(16*mask_repr_power < mask_repr_bytes) {
        mask_repr_power <<= 1;
    }
    uint8_t mask_repr_switch = 32-((mask_repr_bytes-(16*(mask_repr_power>>1)))*2); // all bits <switch use power/2 instead
    if (blockcount == 0) {
        blockcount = (chunk_count/threadcount)+1;
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_4pass_popc_iov_monolithic<<<blockcount, threadcount>>>(d_mask, d_pss, d_iov, chunk_length32, chunk_count, mask_repr_power, mask_repr_switch))
        );
    } else {
        CUDA_TIME(ce_start, ce_stop, 0, &time,
            (kernel_4pass_popc_iov_striding<<<blockcount, threadcount>>>(d_mask, d_pss, d_iov, chunk_length32, chunk_count, mask_repr_power, mask_repr_switch))
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

__global__ void kernel_4pass_pss_gmem_monolithic(uint32_t* pss, uint8_t depth, uint32_t chunk_count, uint32_t* out_count)
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

__global__ void kernel_4pass_pss_gmem_striding(uint32_t* pss, uint8_t depth, uint32_t chunk_count, uint32_t* out_count)
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

float launch_4pass_pss_gmem(
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
                (kernel_4pass_pss_gmem_monolithic<<<blockcount, threadcount>>>(d_pss, i, chunk_count, d_out_count))
            );
            time += ptime;
        }
        // last pass forces result into d_out_count
        CUDA_TIME(ce_start, ce_stop, 0, &ptime, 
            (kernel_4pass_pss_gmem_monolithic<<<1, 1>>>(d_pss, static_cast<uint8_t>(max_depth), chunk_count, d_out_count))
        );
        time += ptime;
    } else {
        for (int i = 0; i < max_depth; i++) {
            uint32_t req_blockcount = ((chunk_count>>i)/(threadcount*2))+1;
            if (blockcount > req_blockcount) {
                blockcount = req_blockcount;
            }
            CUDA_TIME(ce_start, ce_stop, 0, &ptime,
                (kernel_4pass_pss_gmem_striding<<<blockcount, threadcount>>>(d_pss, i, chunk_count, d_out_count))
            );
            time += ptime;
        }
        // last pass forces result into d_out_count
        CUDA_TIME(ce_start, ce_stop, 0, &ptime,
            (kernel_4pass_pss_gmem_monolithic<<<1, 1>>>(d_pss, static_cast<uint8_t>(max_depth), chunk_count, d_out_count))
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
__global__ void kernel_4pass_proc_none_monolithic(
    T* input,
    T* output,
    uint8_t* mask,
    uint32_t* pss,
    uint32_t chunk_length8,
    uint32_t chunk_count,
    uint32_t chunk_count_p2)
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
    uint32_t element_idx = tid*chunk_length8;
    for (uint32_t i = element_idx; i < element_idx+chunk_length8; i++) {
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
__global__ void kernel_4pass_proc_none_striding(
    T* input,
    T* output,
    uint8_t* mask,
    uint32_t* pss,
    uint32_t chunk_length8,
    uint32_t chunk_count,
    uint32_t chunk_count_p2)
{
    for (uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        uint32_t out_idx;
        if (complete_pss) {
            out_idx = pss[tid];
        } else {
            out_idx = d_4pass_pproc_pssidx(tid, pss, chunk_count_p2);
        }
        uint32_t element_idx = tid*chunk_length8;
        for (uint32_t i = element_idx; i < element_idx+chunk_length8; i++) {
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

// processing (for complete and partial pss) with no usage of the iov
template <typename T>
float launch_4pass_proc_none(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    T* d_input,
    T* d_output,
    uint8_t* d_mask,
    uint32_t* d_pss,
    bool full_pss,
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
        if (full_pss) {
            CUDA_TIME(ce_start, ce_stop, 0, &time,
                (kernel_4pass_proc_none_monolithic<T, true><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length/8, chunk_count, 0))
            );
        } else {
            CUDA_TIME(ce_start, ce_stop, 0, &time,
                (kernel_4pass_proc_none_monolithic<T, false><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length/8, chunk_count, chunk_count_p2))
            );
        }
    } else {
        if (full_pss) {
            CUDA_TIME(ce_start, ce_stop, 0, &time,
                (kernel_4pass_proc_none_striding<T, true><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length/8, chunk_count, 0))
            );
        } else {
            CUDA_TIME(ce_start, ce_stop, 0, &time,
                (kernel_4pass_proc_none_striding<T, false><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, chunk_length/8, chunk_count, chunk_count_p2))
            );
        }
    }
    return time;
}

template <typename T, bool complete_pss>
__global__ void kernel_4pass_proc_simple_monolithic(
    T* input,
    T* output,
    uint8_t* mask,
    uint32_t* pss,
    iovRow* iov,
    uint32_t chunk_length8,
    uint32_t chunk_count,
    uint32_t chunk_count_p2)
{
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= chunk_count) {
        return;
    }
    iovRow crow = iov[tid];
    uint32_t out_idx;
    if (complete_pss) {
        out_idx = pss[crow.chunk_id];
    } else {
        out_idx = d_4pass_pproc_pssidx(crow.chunk_id, pss, chunk_count_p2);
    }
    uint32_t element_idx = crow.chunk_id*chunk_length8;
    if (crow.mask_repr == 0) {
        return;
    }
    if (crow.mask_repr == chunk_length8*8) {
        for (uint32_t i = element_idx; i < element_idx+(chunk_length8*8); i++) {
            output[out_idx++] = input[i];
        }
        return;
    }
    for (uint32_t i = element_idx; i < element_idx+chunk_length8; i++) {
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
__global__ void kernel_4pass_proc_simple_striding(
    T* input,
    T* output,
    uint8_t* mask,
    uint32_t* pss,
    iovRow* iov,
    uint32_t chunk_length8,
    uint32_t chunk_count,
    uint32_t chunk_count_p2)
{
    for (uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; tid < chunk_count; tid += blockDim.x * gridDim.x) {
        iovRow crow = iov[tid];
        uint32_t out_idx;
        if (complete_pss) {
            out_idx = pss[crow.chunk_id];
        } else {
            out_idx = d_4pass_pproc_pssidx(crow.chunk_id, pss, chunk_count_p2);
        }
        uint32_t element_idx = crow.chunk_id*chunk_length8;
        if (crow.mask_repr == 0) {
            continue;
        }
        if (crow.mask_repr == chunk_length8*8) {
            for (uint32_t i = element_idx; i < element_idx+(chunk_length8*8); i++) {
                output[out_idx++] = input[i];
            }
            continue;
        }
        for (uint32_t i = element_idx; i < element_idx+chunk_length8; i++) {
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

// processing (for complete and partial pss) with simple usage of the iov
template <typename T>
float launch_4pass_proc_simple(
    cudaEvent_t ce_start,
    cudaEvent_t ce_stop,
    uint32_t blockcount,
    uint32_t threadcount,
    T* d_input,
    T* d_output,
    uint8_t* d_mask,
    uint32_t* d_pss,
    bool full_pss,
    iovRow* d_iov,
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
        if (full_pss) {
            CUDA_TIME(ce_start, ce_stop, 0, &time,
                (kernel_4pass_proc_simple_monolithic<T, true><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, d_iov, chunk_length/8, chunk_count, 0))
            );
        } else {
            CUDA_TIME(ce_start, ce_stop, 0, &time,
                (kernel_4pass_proc_simple_monolithic<T, false><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, d_iov, chunk_length/8, chunk_count, chunk_count_p2))
            );
        }
    } else {
        if (full_pss) {
            CUDA_TIME(ce_start, ce_stop, 0, &time,
                (kernel_4pass_proc_simple_striding<T, true><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, d_iov, chunk_length/8, chunk_count, 0))
            );
        } else {
            CUDA_TIME(ce_start, ce_stop, 0, &time,
                (kernel_4pass_proc_simple_striding<T, false><<<blockcount, threadcount>>>(d_input, d_output, d_mask, d_pss, d_iov, chunk_length/8, chunk_count, chunk_count_p2))
            );
        }
    }
    return time;
}


#endif
