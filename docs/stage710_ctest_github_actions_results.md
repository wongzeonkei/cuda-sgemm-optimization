# Stage 7.10 CTest and GitHub Actions CI

## Objective

Stage 7.10 introduces automated testing through CTest and a GitHub
Actions workflow.

The test architecture separates compile-only package validation from
GPU runtime validation.

## Test Categories

Two CTest label groups are defined:

| Label | Purpose | GPU Required |
|---|---|---|
| `ci` | Build, install, find_package, Consumer and relocation validation | No |
| `gpu` | Execute optimized SGEMM dispatcher kernels | Yes |

GPU tests are controlled with:

```cmake
SGEMM_ENABLE_GPU_TESTS
```

The default value is `OFF`, allowing package tests to run on systems
without an NVIDIA GPU.

## Registered Tests

With GPU testing enabled, CTest registers:

```text
stage710_package_consumer_build
stage710_gpu_float4
stage710_gpu_dimension_fallback
stage710_gpu_pointer_fallback
```

With GPU testing disabled, only the compile-only package test is
registered.

## Compile-Only Package Test

The package CTest performs the following sequence:

1. Installs the SGEMM static library and CMake Config package.
2. Configures an external Consumer with `find_package`.
3. Builds the original Consumer.
4. Copies the complete installation tree to a new directory.
5. Configures a Consumer from the relocated package.
6. Builds the relocated Consumer.
7. Checks that the relocated Consumer links the relocated static library.
8. Rejects references to the original installation path.

The final marker is:

```text
CI_PACKAGE_TEST_PASS = true
```

## Local Makefiles Validation

The local GPU-enabled configuration used:

```bash
cmake \
  -S . \
  -B build-stage710 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_TESTING=ON \
  -DSGEMM_ENABLE_GPU_TESTS=ON
```

Four tests were registered.

The compile-only package test completed with:

```text
100% tests passed, 0 tests failed out of 1
CI_PACKAGE_TEST_PASS = true
```

## GPU Runtime Validation

Three runtime cases were executed on the local RTX 3090.

| Test | Selected Kernel | Mismatches |
|---|---|---:|
| float4-compatible dimensions and aligned pointers | float4 double buffer | 0 |
| dimension-incompatible input | scalar single buffer | 0 |
| misaligned A pointer | scalar single buffer | 0 |

CTest reported:

```text
100% tests passed, 0 tests failed out of 3
```

## Ninja CI Simulation

The GitHub Actions configuration was reproduced locally with:

```bash
cmake \
  -S . \
  -B build-stage710-ci \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_TESTING=ON \
  -DSGEMM_ENABLE_GPU_TESTS=OFF
```

The Ninja build completed all 20 build steps.

Only one test was registered:

```text
stage710_package_consumer_build
```

The Ninja compile-only test completed with:

```text
Original package Consumer build passed.
Relocated package Consumer build passed.
CI_PACKAGE_TEST_PASS = true
100% tests passed, 0 tests failed out of 1
```

## GitHub Actions Design

The workflow performs:

- CUDA development-container initialization
- CMake and Ninja installation
- CUDA 11.8 toolchain reporting
- Release configuration
- static-library compilation
- CTest discovery
- compile-only `ci` label execution
- install-tree validation

The workflow explicitly configures:

```cmake
BUILD_TESTING=ON
SGEMM_ENABLE_GPU_TESTS=OFF
```

GitHub-hosted execution therefore validates compilation and packaging
without attempting to launch CUDA kernels.

GPU runtime tests remain part of the local RTX 3090 validation workflow.

## Compiler Environment

Local validation used:

- CUDA 11.8.89
- GNU C++ 11.5
- CMake Release configuration
- compute capability 8.6
- Unix Makefiles and Ninja generators

## Conclusion

Stage 7.10 successfully introduces automated test registration and
compile-only continuous integration for the SGEMM project.

The package, external Consumer, relocation behavior, and imported CMake
target can now be checked automatically without GPU access, while local
CTest continues to verify the actual float4 and scalar CUDA runtime paths.
