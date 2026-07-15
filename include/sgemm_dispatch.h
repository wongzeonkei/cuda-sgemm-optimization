#pragma once

#include <cuda_runtime.h>

enum class SgemmKernelKind {
    ScalarSingleBuffer,
    Float4DoubleBuffer
};

struct SgemmDispatchInfo {
    SgemmKernelKind kernel;
    bool valid_dimensions;
    bool dimensions_float4_compatible;
    bool pointers_16byte_aligned;
    bool float4_safe;
};

const char* sgemm_kernel_name(
    SgemmKernelKind kernel
);

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
    cudaStream_t stream = nullptr,
    SgemmDispatchInfo* dispatch_info = nullptr
);
