# Stage 7.9 CMake Package Configuration

## Objective

Stage 7.9 packages the reusable SGEMM static library as an installable
CMake Config package.

An external CMake project can discover and link the library through
`find_package` without manually specifying include directories, library
paths, or CUDA Runtime linker flags.

## Package Version

The installed package version is:

```text
0.1.0
```

The package version file is generated with
`write_basic_package_version_file` using `SameMajorVersion`
compatibility.

## Installed Package Files

The installation contains:

```text
include/sgemm_dispatch.h
include/cuda_check.h
lib/libsgemm_dispatch.a
lib/cmake/sgemm_dispatch/sgemm_dispatchConfig.cmake
lib/cmake/sgemm_dispatch/sgemm_dispatchConfigVersion.cmake
lib/cmake/sgemm_dispatch/sgemm_dispatchTargets.cmake
lib/cmake/sgemm_dispatch/sgemm_dispatchTargets-release.cmake
```

## Config Template

The package configuration performs the following operations:

```cmake
include(CMakeFindDependencyMacro)

find_dependency(CUDAToolkit REQUIRED)

set(
    sgemm_dispatch_VERSION
    "0.1.0"
)

include(
    "${CMAKE_CURRENT_LIST_DIR}/sgemm_dispatchTargets.cmake"
)
```

`find_dependency(CUDAToolkit)` ensures that the imported target
`CUDA::cudart` exists before the exported SGEMM target is loaded.

## Consumer Usage

An external project uses the package with:

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

No manual `include_directories`, `link_directories`, static-library path,
or `-lcudart` option is required.

## Imported Target Properties

The installed package reported:

| Property | Value |
|---|---|
| Package version | 0.1.0 |
| Target name | `cuda_sgemm::sgemm_dispatch` |
| Target type | `STATIC_LIBRARY` |
| Include directory | installed `include` directory |
| Interface dependency | `CUDA::cudart` |

The original installation resolved the include directory to:

```text
/root/cuda-sgemm-optimization/install/include
```

## Link Validation

The external Consumer link response file contained:

```text
/root/cuda-sgemm-optimization/install/lib/libsgemm_dispatch.a
/usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so
```

This confirms that the imported target propagated both the installed
static library and the CUDA Runtime dependency.

## Runtime Validation

The external Consumer evaluated two dispatcher paths.

| Input | Expected Kernel | Selected Kernel | Mismatches |
|---|---|---|---:|
| 128 x 132 x 116 | float4 double buffer | float4 double buffer | 0 |
| 127 x 131 x 113 | scalar single buffer | scalar single buffer | 0 |

The final Consumer result was:

```text
PACKAGE_CONSUMER_PASS = true
```

## Version Compatibility

An exact request for version 0.1.0 succeeded:

```text
EXACT_VERSION_PASS=0.1.0
```

A request for incompatible version 1.0.0 was rejected because the
installed package version is 0.1.0.

This confirms that `sgemm_dispatchConfigVersion.cmake` participates
correctly in package selection.

## Relocation Validation

The complete installation tree was copied from the original installation
directory to:

```text
/tmp/sgemm_dispatch_relocated
```

The relocated package was discovered with:

```bash
cmake \
  -S tests/stage79_cmake_consumer \
  -B build/stage79_relocated_consumer \
  -DCMAKE_PREFIX_PATH=/tmp/sgemm_dispatch_relocated
```

The imported include directory changed to:

```text
/tmp/sgemm_dispatch_relocated/include
```

The Consumer linked:

```text
/tmp/sgemm_dispatch_relocated/lib/libsgemm_dispatch.a
```

It did not reference the original installation directory.

The relocated Consumer also completed with:

```text
PACKAGE_CONSUMER_PASS = true
```

This confirms that the generated package is relocatable.

## Compiler Environment

The validation environment used:

- CUDA 11.8.89
- GNU C++ 11.5
- CUDA architecture 86
- Release build configuration

GCC 11 was explicitly selected as the CUDA host compiler because the
installed CUDA 11.8 toolchain rejects newer default GNU compiler versions.

## Conclusion

Stage 7.9 successfully converts the reusable SGEMM static library into a
versioned and relocatable CMake Config package.

External CUDA projects can now discover and link the library through
`find_package` and the imported target
`cuda_sgemm::sgemm_dispatch`.
