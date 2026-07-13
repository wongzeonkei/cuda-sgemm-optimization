# Stage 7.2 Shared Memory Tiling Results

## Configuration

- GPU: NVIDIA GeForce RTX 3090
- CUDA: 11.8
- Data type: FP32
- Matrix layout: Row-major
- Block tile: 32 x 32
- Thread block: 32 x 32, 1024 threads
- Shared memory: 8 KiB per block
- Timing: CUDA Event
- Reference: cuBLAS Pedantic FP32

## Performance

| Matrix Size | Naive GFLOPS | Tiled GFLOPS | Speedup | Tiled/cuBLAS |
|---|---:|---:|---:|---:|
| 512³ | 1995.007 | 2142.049 | 1.074x | 20.363% |
| 1024³ | 2105.363 | 2520.737 | 1.197x | 15.409% |
| 4096³ | 2042.246 | 2768.914 | 1.356x | 14.379% |
| 1000x1030x777 | 2078.598 | 2449.753 | 1.179x | 18.133% |

## Correctness

All tested configurations produced zero mismatches under the configured
absolute and relative tolerances.

The non-tile-aligned 1000x1030x777 case also passed, verifying that zero
padding and output boundary checks work correctly.

## Observation

Shared-memory tiling improves performance by reducing repeated global-memory
loads. The improvement remains limited because each thread still computes
only one output element and the 1024-thread block restricts scheduling
flexibility. Register tiling is the next optimization step.
