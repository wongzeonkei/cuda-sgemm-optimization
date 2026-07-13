# Stage 7: CUDA SGEMM Optimization

## Goal

Implement FP32 SGEMM from scratch and optimize it progressively toward
cuBLAS performance.

## Correctness Requirements

Each CUDA kernel must be compared with a CPU or cuBLAS reference.

Recorded metrics:

- Maximum absolute error
- Mean absolute error
- Relative error
- Pass/fail status

## Benchmark Requirements

- CUDA Event timing
- Warmup before measurement
- Multiple measured iterations
- Same M/N/K for all implementations
- H2D and D2H excluded from kernel timing
- FP32 computation
- Identical hardware and clock conditions

## Planned Matrix Sizes

- 512 x 512 x 512
- 1024 x 1024 x 1024
- 2048 x 2048 x 2048
- 4096 x 4096 x 4096
- Non-square and non-tile-aligned cases

## Optimization Versions

- V0: CPU reference
- V1: cuBLAS reference
- V2: Naive CUDA kernel
- V3: Coalesced global-memory access
- V4: Shared-memory tiling
- V5: Register tiling
- V6: Vectorized float4 load
- V7: Double buffering
