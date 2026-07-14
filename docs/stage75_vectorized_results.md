# Stage 7.5 float4 Vectorized Loading Results

## Configuration

- GPU: NVIDIA GeForce RTX 3090
- CUDA: 11.8
- Data type: FP32
- Matrix layout: Row-major
- Block tile: 128 x 128
- K tile: 8
- Thread tile: 8 x 8
- Threads per block: 256
- Vector width: 4 FP32 values / 16 bytes
- Timing: CUDA Event
- Reference: cuBLAS Pedantic FP32

## Resource Usage

| Kernel | Registers/Thread | Shared Memory | Local Memory | Spill |
|---|---:|---:|---:|---:|
| Scalar Load | 122 | 8320 bytes | 0 | 0 |
| float4 Load | 110 | 8320 bytes | 0 | 0 |

The float4 implementation reduced register usage from 122 to 110 registers
per thread without introducing local-memory usage or register spilling.

## Performance

| Matrix Size | Scalar GFLOPS | Selected GFLOPS | Speedup | Selected/cuBLAS |
|---|---:|---:|---:|---:|
| 512³ | 2347.278 | 2293.875 | 0.977x | 21.752% |
| 1024³ | 7080.593 | 6194.329 | 0.875x | 41.405% |
| 4096³ | 14194.298 | 15830.643 | 1.115x | 72.364% |
| 1000x1032x780 | 8761.959 | 6560.805 | 0.749x | 44.749% |
| 1000x1030x777 | 8842.761 | 8811.192 | 0.996x | 66.291% |

## 4096³ Result

For the aligned 4096³ matrix:

- Scalar-load Register2D: 14.194 TFLOPS
- float4-load Register2D: 15.831 TFLOPS
- Speedup: 1.115x
- cuBLAS ratio: 72.364%
- Mean latency reduction: approximately 10.34%

## Correctness

All tested configurations produced zero mismatches.

The float4 implementation produced exactly the same output as the scalar
Register2D kernel under the tested cases.

Compute Sanitizer results:

- float4 path: ERROR SUMMARY: 0 errors
- scalar fallback path: ERROR SUMMARY: 0 errors

## Observation

float4 loading improves performance for large, aligned matrices, but does
not provide a universal speedup.

The 1024³ and boundary-heavy aligned case were slower than scalar loading.
This shows that safe vectorization and profitable vectorization are separate
conditions.

A production kernel dispatcher should consider matrix size, alignment and
tile utilization rather than selecting float4 based only on pointer and row
alignment.
