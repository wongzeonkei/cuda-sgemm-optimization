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

## Reusable Static Library

The final SGEMM dispatcher is also available as a reusable static library:

```text
build/lib/libsgemm_dispatch.a
```

Build and install it with:

```bash
./scripts/build.sh
cmake --install build --prefix "$(pwd)/install"
```

The installed files include:

```text
install/include/sgemm_dispatch.h
install/include/cuda_check.h
install/lib/libsgemm_dispatch.a
```

Example external compilation for CUDA 11.8:

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
  -o /tmp/sgemm_library_smoke
```

The public API supports row-major FP32 `C = A x B` and accepts an optional
CUDA stream. Vector-compatible inputs use the float4 double-buffer kernel;
all other inputs automatically use the scalar fallback.

### CMake Package Usage

Install the package:

```bash
./scripts/build.sh
cmake --install build --prefix "$(pwd)/install"
```

Use it from an external CMake project:

```cmake
find_package(
    sgemm_dispatch
    0.1
    CONFIG
    REQUIRED
)

target_link_libraries(
    external_application
    PRIVATE
    cuda_sgemm::sgemm_dispatch
)
```

Configure the external project with:

```bash
cmake \
  -S external_project \
  -B external_build \
  -DCMAKE_PREFIX_PATH=/path/to/sgemm/install
```

The imported target automatically propagates the installed header path
and the `CUDA::cudart` dependency. The package is relocatable.

## Automated Testing

Configure compile-only tests:

```bash
cmake \
  -S . \
  -B build-ci \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DSGEMM_ENABLE_GPU_TESTS=OFF

cmake --build build-ci --parallel 2

ctest \
  --test-dir build-ci \
  --label-regex ci \
  --output-on-failure
```

Enable local GPU runtime tests with:

```bash
cmake \
  -S . \
  -B build-gpu-tests \
  -DBUILD_TESTING=ON \
  -DSGEMM_ENABLE_GPU_TESTS=ON

cmake --build build-gpu-tests

ctest \
  --test-dir build-gpu-tests \
  --label-regex gpu \
  --output-on-failure
```

The compile-only test installs the CMake package, builds an external
Consumer, relocates the installation tree, and builds the Consumer again.
GPU tests validate the float4 kernel and scalar fallback paths.
