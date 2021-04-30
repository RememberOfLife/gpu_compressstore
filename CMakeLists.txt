cmake_minimum_required(VERSION 3.17)
project(gpu_compressstore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(INCLUDES
    src
)

set(SOURCES
    src/main.cu

    src/avx_wrap.cuh
    src/benchmark_data.cuh
    src/cub_wraps.cuh
    src/cuda_time.cuh
    src/cuda_try.cuh
    src/fast_prng.cpp
    src/fast_prng.hpp

    src/kernels/kernel_3pass.cuh
    src/kernels/kernel_copy_add.cuh
    src/kernels/kernel_singlethread.cuh
)

add_executable(gpu_compressstore ${SOURCES})

option(AVXPOWER "use avx512-ish" OFF)
if(AVXPOWER)
    target_compile_definitions(gpu_compressstore PUBLIC AVXPOWER)
    set(CMAKE_CUDA_FLAGS "-Xcompiler -march=native")
endif()

set_property(TARGET gpu_compressstore PROPERTY CUDA_ARCHITECTURES 75 61)

target_include_directories(gpu_compressstore PRIVATE ${INCLUDES})
