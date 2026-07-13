# Stage 7.4 2D Register Tiling Results

## Configuration

- GPU: NVIDIA GeForce RTX 3090
- CUDA: 11.8
- Data type: FP32
- Matrix layout: Row-major
- Block tile: 128 x 128
- K tile: 8
- Thread tile: 8 x 8
- Outputs per thread: 64
- Threads per block: 256
- Registers per thread: 124
- Static shared memory: 8320 bytes
- Local memory: 0 bytes
- Spill loads/stores: 0
- Timing: CUDA Event
- Reference: cuBLAS Pedantic FP32

## Performance

| Matrix Size | 2D Register GFLOPS | cuBLAS GFLOPS | 2D/cuBLAS |
|---|---:|---:|---:|
| 512³ | 2275.555 | 10570.323 | 21.528% |
| 1024³ | 7002.177 | 15258.673 | 45.890% |
| 4096³ | 14294.297 | 22419.880 | 63.757% |
| 1000x1030x777 | 7140.729 | 15815.570 | 45.150% |

## Optimization Evolution at 4096³

| Kernel | GFLOPS | Speedup over Naive |
|---|---:|---:|
| Naive CUDA | 2042.246 | 1.000x |
| Shared Memory Tiling | 2768.914 | 1.356x |
| 1D Register Tiling | 6863.196 | 3.361x |
| 2D Register Tiling | 14294.297 | 6.999x |
| cuBLAS Pedantic FP32 | 22419.880 | 10.978x |

The 2D register-tiled kernel is approximately 2.083x faster than the
1D register-tiled kernel at 4096³.

## Resource Usage

The compiler reported:

- 124 registers per thread
- 8320 bytes static shared memory per block
- 0 bytes spill stores
- 0 bytes spill loads
- 0 bytes local memory

Register usage limits the kernel to approximately two 256-thread blocks per
SM, corresponding to a theoretical occupancy of about 33 percent. Despite
the relatively low occupancy, high register-level data reuse provides strong
performance.

## Correctness

All tested matrix sizes produced zero mismatches under the configured
absolute and relative tolerances.

The non-aligned 1000x1030x777 test passed.

Compute Sanitizer reported:

- ERROR SUMMARY: 0 errors

## Small-Matrix Observation

The 512³ case is slower than the 1D register-tiled version because the
128x128 block tile generates only 16 thread blocks. This is insufficient to
occupy the 82 SMs of the RTX 3090. The large-tile kernel is primarily
effective for medium and large matrices.
