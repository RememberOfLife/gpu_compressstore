#ifndef BENCHMARK_DATA_CUH
#define BENCHMARK_DATA_CUH

#include <cstdint>

#include "benchmark_data.cuh"

enum MaskType {
    MASKTYPE_UNIFORM = 0, // every bit is decided uniformly random
    MASKTYPE_ZIPF = 1, // lots of 1s in the front thinning out quickly towards the back
    MASKTYPE_BURST = 2, // randomly placed bursts of 1s
    MASKTYPE_OFFSET = 3, // 1s with gaps of 0s: 1=101010..; 2=1001001..
};

typedef struct benchmark_data {
    bool validation;
    uint64_t size; // elems
    uint64_t* h_input;
    uint8_t* h_mask;
    uint64_t* h_validation; // correct results
    uint64_t* h_output; // gpu result buffer
    uint64_t* d_input;
    uint8_t* d_mask;
    uint64_t* d_output;
    cudaEvent_t ce_start;
    cudaEvent_t ce_stop;

    benchmark_data(bool validation, uint64_t size);
    ~benchmark_data();
    void generate_mask(MaskType mtype, double marg);
    bool validate(uint64_t count);
} benchmark_data;

#endif
