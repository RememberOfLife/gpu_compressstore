cmake_minimum_required(VERSION 3.8)
project(gpu_compressstore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

set(INCLUDES
    src
)

set(SOURCES
    src/main.cu

    src/benchmark_data.cuh
    src/cub_wraps.cuh
    src/cuda_time.cuh
    src/cuda_try.cuh
    src/fast_prng.cpp
    src/fast_prng.hpp

    src/kernels/kernel_4pass.cuh
    src/kernels/kernel_singlethread.cuh
)

add_executable(gpu_compressstore ${SOURCES})
target_include_directories(gpu_compressstore PRIVATE ${INCLUDES})
