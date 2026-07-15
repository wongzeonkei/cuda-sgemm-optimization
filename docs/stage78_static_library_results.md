# Stage 7.8 Reusable SGEMM Static Library

## Objective

Stage 7.8 separates the optimized CUDA kernels and runtime dispatcher
from the Stage 7.7 benchmark executable and packages them as a reusable
static library.

## Module Layout

```text
include/sgemm_dispatch.h
src/sgemm_dispatch.cu
tests/sgemm_stage78_library_smoke.cu
build/lib/libsgemm_dispatch.a
```

The Stage 7.7 executable remains unchanged as a historical benchmark
implementation.

## Public API

The library exports:

```cpp
const char* sgemm_kernel_name(SgemmKernelKind kernel);

SgemmDispatchInfo select_sgemm_kernel(
    const float* d_A,
    const float* d_B,
    int M,
    int N,
    int K
);

cudaError_t launch_sgemm_dispatch(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    SgemmDispatchInfo* dispatch_info
);
```

The library does not contain a benchmark entry point, cuBLAS calls, or
a `main()` function.

## Dispatch Policy

The float4 double-buffer kernel is selected when:

- N is divisible by 4
- K is divisible by 4
- A is aligned to 16 bytes
- B is aligned to 16 bytes

All other inputs use the scalar single-buffer kernel.

## Kernel Resources

| Kernel | Registers/Thread | Shared Memory | Local Memory | Spill |
|---|---:|---:|---:|---:|
| float4 double buffer | 128 | 16640 bytes | 0 | 0 |
| scalar single buffer | 116 | 8320 bytes | 0 | 0 |

The module extraction did not change the compiled kernel resources.

## Static Library Validation

The archive was generated at:

```text
build/lib/libsgemm_dispatch.a
```

The installed archive was successfully linked into a separate executable
using the installed headers and library.

External compilation required GCC 11 because CUDA 11.8 rejects newer
default GNU host compilers:

```bash
nvcc \
  -std=c++17 \
  -arch=sm_86 \
  -O3 \
  -ccbin /usr/bin/g++-11 \
  -Iinstall/include \
  tests/sgemm_stage78_library_smoke.cu \
  install/lib/libsgemm_dispatch.a \
  -lcudart \
  -o /tmp/sgemm_stage78_external_smoke
```

## Functional Tests

| Test | Selected Kernel | Max Abs Error | Mismatches |
|---|---|---:|---:|
| 1024x1024x1024 aligned | float4 double buffer | 0 | 0 |
| 1000x1030x777 aligned | scalar single buffer | 0 | 0 |
| 127x132x116 misaligned A | scalar single buffer | 0 | 0 |
| 127x132x116 misaligned B | scalar single buffer | 0 | 0 |
| external 512x512x512 | float4 double buffer | 0 | 0 |

## Symbol Audit

The static library contains the three public dispatcher functions:

- `sgemm_kernel_name`
- `select_sgemm_kernel`
- `launch_sgemm_dispatch`

It does not contain:

- `main`
- `cublasSgemm`
- `benchmark_dispatch`
- `benchmark_cublas`

## Compute Sanitizer

The following paths were checked with Compute Sanitizer memcheck:

- aligned float4 double-buffer path
- dimension-based scalar fallback
- pointer-alignment-based scalar fallback

All three tests completed with:

```text
Mismatches = 0
ERROR SUMMARY: 0 errors
```

## Performance Interpretation

The static-library smoke tests use all-one matrices and are intended to
validate linking, dispatch, and numerical correctness. They are not used
as the official performance benchmark.

The official repeated Stage 7.7 result remains:

- 4096-cubed median: 17.642 TFLOPS
- best observed result: 18.015 TFLOPS
- median ratio to cuBLAS Pedantic FP32: 82.049%

## Current Interface Scope

The public implementation currently supports:

- FP32
- row-major A, B, and C
- A shape M x K
- B shape K x N
- C shape M x N
- C = A x B
- optional CUDA stream execution

It does not yet expose alpha, beta, transpose operations, or alternative
data types.

## Conclusion

Stage 7.8 successfully converts the final SGEMM dispatcher into a
standalone static library that can be installed and linked by an external
CUDA application.
