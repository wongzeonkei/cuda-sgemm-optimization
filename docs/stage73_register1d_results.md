# Stage 7.3 1D Register Tiling Results

## Configuration

- GPU: NVIDIA GeForce RTX 3090
- CUDA: 11.8
- Data type: FP32
- Matrix layout: Row-major
- Block tile: 128 x 32
- K tile: 16
- Thread tile: 8 x 1
- Threads per block: 512
- Registers per thread: 64
- Shared memory per block: 10240 bytes
- Register spill loads/stores: 0
- Timing: CUDA Event
- Reference: cuBLAS Pedantic FP32

## Performance

| Matrix Size | Register Tiled GFLOPS | cuBLAS GFLOPS | Register/cuBLAS |
|---|---:|---:|---:|
| 512³ | 4369.067 | 10208.100 | 42.800% |
| 1024³ | 5530.464 | 14741.684 | 37.516% |
| 4096³ | 6863.196 | 22255.562 | 30.838% |
| 1000x1030x777 | 5528.551 | 15804.908 | 34.980% |

## Optimization Effect

For the 4096³ matrix:

- Naive CUDA: 2042.246 GFLOPS
- Shared-memory tiled: 2768.914 GFLOPS
- 1D register tiled: 6863.196 GFLOPS

The register-tiled kernel is approximately:

- 2.479x faster than the shared-memory tiled kernel
- 3.361x faster than the naive CUDA kernel

## Correctness

All tested matrix sizes produced zero mismatches under the configured
absolute and relative tolerances.

Compute Sanitizer reported:

- ERROR SUMMARY: 0 errors

The compiler reported:

- 64 registers per thread
- 10240 bytes shared memory per block
- 0 spill stores
- 0 spill loads

## Observation

Each thread computes eight output values in one output column. This allows
one B value to be reused across eight register accumulators. The next step
is two-dimensional register tiling, where each thread computes an 8 x 8
output tile and reuses both A and B values.
