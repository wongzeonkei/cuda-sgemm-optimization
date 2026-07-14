# Stage 7.7 Unified Benchmark Results

## Protocol

- Repeats per executable: 5
- Statistic used for dispatch decisions: median
- GPU timing: CUDA Events inside each executable
- Reference: cuBLAS Pedantic FP32
- Execution order alternates between Stage 7.5 and Stage 7.6
- Raw logs and CSV files are stored under `results/stage77_runs/`

## aligned_edge

Shape: 1000 x 1032 x 780

| Source | Implementation | Runs | Median ms | Median GFLOPS | Min GFLOPS | Max GFLOPS | CV | cuBLAS ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage75 | cublas_pedantic_fp32 | 5 | 0.100983 | 15942.400 | 15931.600 | 15953.200 | 0.054% | 100.000% |
| stage76 | cublas_pedantic_fp32 | 5 | 0.101018 | 15937.000 | 14568.500 | 15955.900 | 3.828% | 100.000% |
| stage76 | double_buffer_register2d | 5 | 0.147849 | 10889.000 | 10807.900 | 11600.000 | 3.729% | 68.325% |
| stage75 | register2d_scalar_load | 5 | 0.183808 | 8758.700 | 7189.880 | 8766.030 | 7.774% | 54.940% |
| stage76 | single_buffer_register2d | 5 | 0.196369 | 8198.440 | 7825.070 | 8611.580 | 3.399% | 51.443% |
| stage75 | register2d_float4_load | 5 | 0.239070 | 6734.100 | 6519.990 | 6795.220 | 2.019% | 42.240% |

**Best custom implementation:** `stage76:double_buffer_register2d` at 10889.000 GFLOPS.

## scalar_fallback

Shape: 1000 x 1030 x 777

| Source | Implementation | Runs | Median ms | Median GFLOPS | Min GFLOPS | Max GFLOPS | CV | cuBLAS ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage76 | cublas_pedantic_fp32 | 5 | 0.101325 | 15796.900 | 14455.400 | 15807.600 | 4.483% | 100.000% |
| stage75 | cublas_pedantic_fp32 | 5 | 0.101632 | 15749.200 | 14459.800 | 15802.200 | 3.739% | 100.000% |
| stage76 | single_buffer_register2d | 5 | 0.171793 | 9317.140 | 8851.940 | 9327.330 | 2.529% | 58.981% |
| stage76 | double_buffer_register2d | 5 | 0.180275 | 8878.760 | 8479.010 | 9341.270 | 4.102% | 56.206% |
| stage75 | register2d_scalar_load | 5 | 0.181367 | 8825.290 | 8374.530 | 8833.600 | 2.296% | 56.036% |
| stage75 | register2d_scalar_fallback | 5 | 0.190805 | 8388.760 | 8233.370 | 8830.270 | 3.309% | 53.265% |

**Best custom implementation:** `stage76:single_buffer_register2d` at 9317.140 GFLOPS.

## square_1024

Shape: 1024 x 1024 x 1024

| Source | Implementation | Runs | Median ms | Median GFLOPS | Min GFLOPS | Max GFLOPS | CV | cuBLAS ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage75 | cublas_pedantic_fp32 | 5 | 0.135219 | 15881.500 | 14513.500 | 16435.400 | 4.467% | 100.000% |
| stage76 | cublas_pedantic_fp32 | 5 | 0.137318 | 15638.700 | 12751.600 | 15992.000 | 8.683% | 100.000% |
| stage76 | double_buffer_register2d | 5 | 0.172787 | 12428.500 | 10832.400 | 12447.900 | 5.633% | 79.473% |
| stage76 | single_buffer_register2d | 5 | 0.240986 | 8911.250 | 8551.510 | 9156.870 | 2.788% | 56.982% |
| stage75 | register2d_scalar_load | 5 | 0.260390 | 8247.170 | 8230.090 | 8469.920 | 1.219% | 51.929% |
| stage75 | register2d_float4_load | 5 | 0.294784 | 7284.940 | 7099.060 | 7450.910 | 1.984% | 45.871% |

**Best custom implementation:** `stage76:double_buffer_register2d` at 12428.500 GFLOPS.

## square_1536

Shape: 1536 x 1536 x 1536

| Source | Implementation | Runs | Median ms | Median GFLOPS | Min GFLOPS | Max GFLOPS | CV | cuBLAS ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage75 | cublas_pedantic_fp32 | 5 | 0.327311 | 22143.300 | 19660.800 | 22526.700 | 5.364% | 100.000% |
| stage76 | cublas_pedantic_fp32 | 5 | 0.342835 | 21140.600 | 20544.200 | 21830.800 | 2.445% | 100.000% |
| stage76 | single_buffer_register2d | 5 | 0.447304 | 16203.200 | 15356.000 | 17108.500 | 5.156% | 76.645% |
| stage76 | double_buffer_register2d | 5 | 0.461046 | 15720.300 | 14882.000 | 16628.800 | 4.444% | 74.361% |
| stage75 | register2d_float4_load | 5 | 0.495227 | 14635.200 | 12508.100 | 14910.900 | 6.814% | 66.093% |
| stage75 | register2d_scalar_load | 5 | 0.593080 | 12220.500 | 11559.900 | 13497.600 | 7.008% | 55.188% |

**Best custom implementation:** `stage76:single_buffer_register2d` at 16203.200 GFLOPS.

## square_2048

Shape: 2048 x 2048 x 2048

| Source | Implementation | Runs | Median ms | Median GFLOPS | Min GFLOPS | Max GFLOPS | CV | cuBLAS ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage75 | cublas_pedantic_fp32 | 5 | 0.834842 | 20578.600 | 20318.200 | 20923.100 | 1.142% | 100.000% |
| stage76 | cublas_pedantic_fp32 | 5 | 0.835174 | 20570.400 | 20288.700 | 21022.800 | 1.300% | 100.000% |
| stage76 | double_buffer_register2d | 5 | 1.138070 | 15095.600 | 14198.100 | 15307.700 | 2.874% | 73.385% |
| stage76 | single_buffer_register2d | 5 | 1.157430 | 14843.200 | 14209.300 | 14912.100 | 2.016% | 72.158% |
| stage75 | register2d_float4_load | 5 | 1.239810 | 13856.900 | 13301.500 | 14015.800 | 2.232% | 67.336% |
| stage75 | register2d_scalar_load | 5 | 1.404440 | 12232.500 | 12039.800 | 12380.100 | 1.129% | 59.443% |

**Best custom implementation:** `stage76:double_buffer_register2d` at 15095.600 GFLOPS.

## square_3072

Shape: 3072 x 3072 x 3072

| Source | Implementation | Runs | Median ms | Median GFLOPS | Min GFLOPS | Max GFLOPS | CV | cuBLAS ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage75 | cublas_pedantic_fp32 | 5 | 2.727490 | 21258.400 | 20533.500 | 21300.000 | 1.529% | 100.000% |
| stage76 | cublas_pedantic_fp32 | 5 | 2.752060 | 21068.600 | 20223.100 | 21214.800 | 1.916% | 100.000% |
| stage76 | double_buffer_register2d | 5 | 3.460220 | 16756.800 | 16240.600 | 16876.200 | 1.498% | 79.534% |
| stage76 | single_buffer_register2d | 5 | 3.685500 | 15732.500 | 15256.400 | 15836.200 | 1.517% | 74.673% |
| stage75 | register2d_float4_load | 5 | 3.882230 | 14935.200 | 14814.000 | 15028.800 | 0.531% | 70.256% |
| stage75 | register2d_scalar_load | 5 | 4.439860 | 13059.400 | 12691.400 | 13189.600 | 1.554% | 61.432% |

**Best custom implementation:** `stage76:double_buffer_register2d` at 16756.800 GFLOPS.

## square_4096

Shape: 4096 x 4096 x 4096

| Source | Implementation | Runs | Median ms | Median GFLOPS | Min GFLOPS | Max GFLOPS | CV | cuBLAS ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage75 | cublas_pedantic_fp32 | 5 | 6.248090 | 21997.000 | 21677.600 | 22325.500 | 1.064% | 100.000% |
| stage76 | cublas_pedantic_fp32 | 5 | 6.392060 | 21501.500 | 19177.900 | 22032.500 | 5.412% | 100.000% |
| stage76 | double_buffer_register2d | 5 | 7.790590 | 17641.700 | 15376.900 | 17693.800 | 5.662% | 82.049% |
| stage76 | single_buffer_register2d | 5 | 8.216470 | 16727.200 | 16259.300 | 16824.700 | 1.416% | 77.796% |
| stage75 | register2d_float4_load | 5 | 8.886730 | 15465.600 | 15363.200 | 15741.000 | 1.105% | 70.308% |
| stage75 | register2d_scalar_load | 5 | 9.565030 | 14368.900 | 14171.100 | 14424.300 | 0.680% | 65.322% |

**Best custom implementation:** `stage76:double_buffer_register2d` at 17641.700 GFLOPS.

## square_512

Shape: 512 x 512 x 512

| Source | Implementation | Runs | Median ms | Median GFLOPS | Min GFLOPS | Max GFLOPS | CV | cuBLAS ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage76 | cublas_pedantic_fp32 | 5 | 0.026122 | 10276.100 | 9866.160 | 10549.100 | 2.388% | 100.000% |
| stage75 | cublas_pedantic_fp32 | 5 | 0.026235 | 10232.000 | 10009.300 | 10549.100 | 2.172% | 100.000% |
| stage76 | double_buffer_register2d | 5 | 0.090696 | 2959.740 | 2953.420 | 3110.010 | 2.764% | 28.802% |
| stage76 | single_buffer_register2d | 5 | 0.110305 | 2433.570 | 2380.960 | 2491.860 | 2.131% | 23.682% |
| stage75 | register2d_scalar_load | 5 | 0.114442 | 2345.600 | 2242.270 | 2345.810 | 2.412% | 22.924% |
| stage75 | register2d_float4_load | 5 | 0.117115 | 2292.070 | 2191.470 | 2292.670 | 1.981% | 22.401% |

**Best custom implementation:** `stage76:double_buffer_register2d` at 2959.740 GFLOPS.

## Dispatch Decision

The final dispatch policy must be selected from median results,
not from a single historical run. The scalar fallback and
float4-aligned paths should be evaluated separately.
