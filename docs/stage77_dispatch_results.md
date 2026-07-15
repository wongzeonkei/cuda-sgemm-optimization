# Stage 7.7 SGEMM Dispatcher Results

## Objective

Stage 7.7 implements a runtime dispatcher that selects between:

- float4 double-buffered Register2D SGEMM
- scalar single-buffered Register2D SGEMM

The dispatcher selects the optimized float4 kernel only when the matrix
dimensions and device pointers satisfy all vector-access requirements.

## Dispatch Policy

The float4 double-buffered kernel is selected when:

- N is divisible by 4
- K is divisible by 4
- A is aligned to 16 bytes
- B is aligned to 16 bytes

Otherwise, the dispatcher selects the scalar single-buffered kernel.

M does not participate in the vector-access condition because A is
vectorized along K and B is vectorized along N.

## Kernel Resources

| Kernel | Registers/Thread | Shared Memory | Local Memory | Spill |
|---|---:|---:|---:|---:|
| float4 double buffer | 128 | 16640 bytes | 0 | 0 |
| scalar single buffer | 116 | 8320 bytes | 0 | 0 |

## Dispatch Validation

| Shape / Condition | Selected Kernel | Mismatches |
|---|---|---:|
| 1024x1024x1024, aligned | float4 double buffer | 0 |
| 1000x1032x780, aligned | float4 double buffer | 0 |
| 1000x1030x777, aligned | scalar single buffer | 0 |
| 127x132x116, misaligned A | scalar single buffer | 0 |
| 127x132x116, misaligned B | scalar single buffer | 0 |
| 4096x4096x4096, aligned | float4 double buffer | 0 |

## Performance

### 4096 Cubed

- Mean latency: 7.629312 ms
- Performance: 18.014593 TFLOPS
- cuBLAS Pedantic FP32: 21.626917 TFLOPS
- Ratio to cuBLAS: 83.297%
- Mismatches: 0

This is the best observed single-run result.

The repeated Stage 7.7 benchmark remains the official stable result:

- Median custom performance: 17.641700 TFLOPS
- Median ratio to cuBLAS: 82.049%
- Five repeated measurements

### 1024 Cubed

- Mean latency: 0.172524 ms
- Performance: 12.447484 TFLOPS
- Mismatches: 0

The reported 90.254% cuBLAS ratio in this individual run should not be
treated as the stable ratio because cuBLAS performance fluctuated during
that run.

## Compute Sanitizer

The following paths were validated with Compute Sanitizer memcheck:

- aligned float4 double-buffer path
- dimension-based scalar fallback
- pointer-alignment-based scalar fallback

All tests completed with:

```text
ERROR SUMMARY: 0 errors
```

## Conclusion

The final runtime policy is:

```text
float4-compatible dimensions and 16-byte-aligned pointers
    -> float4 double-buffer kernel

otherwise
    -> scalar single-buffer kernel
```

The dispatcher preserves correctness and memory safety for aligned,
unaligned, tile-boundary, and pointer-offset inputs.
