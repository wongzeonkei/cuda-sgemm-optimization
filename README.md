# CUDA SGEMM Optimization

This project implements and optimizes FP32 SGEMM kernels using CUDA C++.

## Optimization Roadmap

1. CPU reference and cuBLAS baseline
2. Naive CUDA SGEMM
3. Global memory coalescing
4. Shared-memory tiling
5. Shared-memory bank-conflict analysis
6. Register tiling
7. Vectorized `float4` loading
8. Double buffering and prefetch
9. Nsight Compute profiling
10. Performance comparison with cuBLAS

## Target Platform

- GPU: NVIDIA GeForce RTX 3090
- Architecture: Ampere
- Compute Capability: 8.6
- Data type: FP32
