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

## Final SGEMM Dispatcher

The final implementation uses a runtime dispatcher with two CUDA kernels:

- **float4 double-buffered Register2D kernel** for vector-compatible inputs
- **scalar single-buffered Register2D kernel** for general fallback inputs

The float4 path requires:

- `N % 4 == 0`
- `K % 4 == 0`
- 16-byte-aligned A and B device pointers

Otherwise, the scalar fallback is selected automatically.

### RTX 3090 FP32 Result

For a row-major `4096 x 4096 x 4096` SGEMM:

| Metric | Result |
|---|---:|
| Five-run median | 17.642 TFLOPS |
| Best observed | 18.015 TFLOPS |
| Median cuBLAS ratio | 82.049% |
| Best latency | 7.629 ms |
| Numerical mismatches | 0 |
| Compute Sanitizer errors | 0 |

The benchmark uses CUDA Events and excludes host-to-device and
device-to-host transfers. The reference implementation is cuBLAS
Pedantic FP32.
