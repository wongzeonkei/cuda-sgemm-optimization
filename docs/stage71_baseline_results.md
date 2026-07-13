# Stage 7.1 SGEMM Baseline Results

## Platform

- GPU: NVIDIA GeForce RTX 3090
- Compute capability: 8.6
- CUDA: 11.8
- Host compiler: G++ 11
- Matrix data type: FP32
- Layout: Row-major
- Timing: CUDA Event
- H2D/D2H excluded
- cuBLAS mode: Pedantic FP32

## Performance

| Matrix Size | Naive Latency | Naive GFLOPS | cuBLAS Latency | cuBLAS GFLOPS | Naive/cuBLAS |
|---|---:|---:|---:|---:|---:|
| 512³ | 0.134390 ms | 1997.440 | 0.025477 ms | 10536.334 | 18.958% |
| 1024³ | 1.096438 ms | 1958.601 | 0.136929 ms | 15683.158 | 12.489% |
| 4096³ | 65.982361 ms | 2082.965 | 6.304768 ms | 21799.208 | 9.555% |

## Correctness

For the 512³ case:

- Naive CUDA vs CPU mismatches: 0
- cuBLAS vs CPU mismatches: 0
- Naive CUDA vs cuBLAS mismatches: 0

The 1024³ and 4096³ cases were validated against cuBLAS and produced zero mismatches under the configured combined absolute and relative tolerance.

## Observation

The naive one-thread-per-output kernel remains near 2 TFLOPS across the tested sizes. cuBLAS scales more effectively with matrix size, reaching approximately 21.8 TFLOPS at 4096³. The performance gap increases because the naive kernel lacks explicit shared-memory and register-level data reuse.
