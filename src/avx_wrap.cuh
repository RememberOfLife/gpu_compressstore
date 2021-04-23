#ifndef AVX_WRAP_CUH
#define AVX_WRAP_CUH

#include <chrono>
#include <cstdint>
#include <immintrin.h>

#include "benchmark_data.cuh"

#ifdef  AVXPOWER

template <typename T>
struct template_type_switch {
    process(T* input, uint8_t* mask, T* output, uint64_t N);
};

template <>
struct template_type_switch<uint8_t> {
    process(uint8_t* input, uint8_t* mask, uint8_t* output, uint64_t N)
    {
        while (input < input+N) {
            // load data and mask
            __m512i a = _mm512_load_epi8(input);
            __mmask64 k = _load_mask64(reinterpret_cast<__mmask8*>(mask));
            // compressstore into output_p
            __mm512_mask_compressstoreu_epi8(output, k, a); // unaligned, no aligned version available
            // increase input_p by processed elems and output_p by popc of mask
            input += 64;
            mask += 8;
            output += _mm_popcnt_u64(k);
        }
    }
};

template <>
struct template_type_switch<uint16_t> {
    process(uint16_t* input, uint8_t* mask, uint16_t* output, uint64_t N)
    {
        while (input < input+N) {
            // load data and mask
            __m512i a = _mm512_load_epi16(input);
            __mmask32 k = _load_mask32(reinterpret_cast<__mmask8*>(mask));
            // compressstore into output_p
            __mm512_mask_compressstoreu_epi16(output, k, a); // unaligned, no aligned version available
            // increase input_p by processed elems and output_p by popc of mask
            input += 32;
            mask += 4;
            output += _mm_popcnt_u64(k);
        }
    }
};

template <>
struct template_type_switch<uint32_t> {
    process(uint32_t* input, uint8_t* mask, uint32_t* output, uint64_t N)
    {
        while (input < input+N) {
            // load data and mask
            __m512i a = _mm512_load_epi32(input);
            __mmask16 k = _load_mask16(reinterpret_cast<__mmask8*>(mask));
            // compressstore into output_p
            __mm512_mask_compressstoreu_epi32(output, k, a); // unaligned, no aligned version available
            // increase input_p by processed elems and output_p by popc of mask
            input += 16;
            mask += 2;
            output += _mm_popcnt_u64(k);
        }
    }
};

template <>
struct template_type_switch<uint64_t> {
    process(uint64_t* input, uint8_t* mask, uint64_t* output, uint64_t N)
    {
        while (input < input+N) {
            // load data and mask
            __m512i a = _mm512_load_epi64(input);
            __mmask8 k = _load_mask8(reinterpret_cast<__mmask8*>(mask));
            // compressstore into output_p
            __mm512_mask_compressstoreu_epi64(output, k, a); // unaligned, no aligned version available
            // increase input_p by processed elems and output_p by popc of mask
            input += 8;
            mask += 1;
            output += _mm_popcnt_u64(k);
        }
    }
};

// hostside avx compressstore wrapper for datatypes
template <typename T>
float launch_avx_compressstore(T* input, uint8_t* mask, T* output, uint64_t N) {
    std::chrono::time_point<std::chrono::steady_clock> start_clock = std::chrono::steady_clock::now();
    template_type_switch<T>::process(input, mask, output, N);
    std::chrono::time_point<std::chrono::steady_clock> stop_clock = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop_clock-start_clock).count();
}

#endif

#endif
