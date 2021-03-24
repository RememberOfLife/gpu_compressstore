## TODO:
* template parameters for different uint sizes (64/32/16)
* fix input sizes to sane multiple (8/32) to accomodate bitmask and kernels
* more bitmask patterns
  * zipf front to back
  * clustered streaks
* logging utility for creating benchmarks

## Optimizations:
useful things from the manual https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

* async copying to and from shared memory is faster (skips RF and L1) https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#async-copy

