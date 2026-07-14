# Stage 7.6 Software Double Buffering Results

## Configuration

- GPU: NVIDIA GeForce RTX 3090
- CUDA: 11.8
- Data type: FP32
- Matrix layout: Row-major
- Block tile: 128 x 128
- K tile: 8
- Thread tile: 8 x 8
- Threads per block: 256
- Global loading: float4 with scalar fallback
- Timing: CUDA Event
- Reference: cuBLAS Pedantic FP32

## Resource Usage

| Kernel | Load Path | Registers/Thread | Shared Memory | Local Memory | Spill |
|---|---|---:|---:|---:|---:|
| Single Buffer | float4 | 110 | 8320 bytes | 0 | 0 |
| Double Buffer | float4 | 128 | 16640 bytes | 0 | 0 |
| Single Buffer | scalar | 116 | 8320 bytes | 0 | 0 |
| Double Buffer | scalar | 116 | 16640 bytes | 0 | 0 |

The double-buffered float4 kernel uses 18 additional registers per thread
and twice the shared-memory capacity. No register spilling or local-memory
allocation was reported.

## Performance

| Matrix Size | Single GFLOPS | Double GFLOPS | Speedup | Double/cuBLAS |
|---|---:|---:|---:|---:|
| 512³ | 2497.041 | 2823.002 | 1.131x | 28.301% |
| 1024³ | 9176.302 | 12488.995 | 1.361x | 83.576% |
| 2048³ | 14375.953 | 15202.720 | 1.058x | 73.114% |
| 4096³ | 14877.540 | 15604.988 | 1.049x | 70.692% |
| 1000x1032x780 | 8597.257 | 11588.606 | 1.348x | 86.242% |
| 1000x1030x777 | 9330.116 | 9318.991 | 0.999x | 68.144% |

## 4096³ Result

For the 4096³ matrix:

- Single-buffer latency: 9.238016 ms
- Double-buffer latency: 8.807373 ms
- Single-buffer performance: 14.878 TFLOPS
- Double-buffer performance: 15.605 TFLOPS
- Same-run speedup: 1.049x
- Double-buffer/cuBLAS ratio: 70.692%

The Stage 7.5 historical float4 result remains slightly higher at
15.831 TFLOPS. Therefore, software double buffering is a valid local
optimization but does not establish a new overall performance record.

## Correctness

All tested configurations produced zero mismatches.

The double-buffer and single-buffer implementations produced identical
outputs in all tested cases.

Compute Sanitizer results:

- float4 double-buffer path: ERROR SUMMARY: 0 errors
- scalar fallback path: ERROR SUMMARY: 0 errors

## Observation

Software double buffering overlaps global-memory prefetch with arithmetic
and reduces the synchronization count from two barriers per K tile to one.

The optimization is effective for the float4 path, particularly for the
1024³ and aligned boundary tests. Its benefit is smaller at 4096³ because
the kernel is increasingly compute-bound and register usage rises from
110 to 128 registers per thread.

The scalar fallback path receives no meaningful benefit. A production
dispatcher should retain the single-buffer scalar kernel for unaligned
matrix dimensions.
