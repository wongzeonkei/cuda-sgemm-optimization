#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "sgemm_dispatch.h"


bool check_cuda(
    cudaError_t status,
    const char* expression,
    const char* file,
    int line
) {
    if (status == cudaSuccess) {
        return true;
    }

    std::cerr
        << "CUDA error: "
        << cudaGetErrorString(status)
        << "\nExpression: "
        << expression
        << "\nLocation: "
        << file
        << ":"
        << line
        << "\n";

    return false;
}


#define CUDA_CHECK(expression)                         \
    do {                                               \
        if (!check_cuda(                               \
                (expression),                          \
                #expression,                           \
                __FILE__,                              \
                __LINE__)) {                           \
            return false;                              \
        }                                              \
    } while (false)


bool run_case(
    int M,
    int N,
    int K,
    const std::string& expected_kernel
) {
    const std::size_t elements_A =
        static_cast<std::size_t>(M) * K;

    const std::size_t elements_B =
        static_cast<std::size_t>(K) * N;

    const std::size_t elements_C =
        static_cast<std::size_t>(M) * N;

    const std::size_t bytes_A =
        elements_A * sizeof(float);

    const std::size_t bytes_B =
        elements_B * sizeof(float);

    const std::size_t bytes_C =
        elements_C * sizeof(float);

    std::vector<float> h_A(elements_A, 1.0f);
    std::vector<float> h_B(elements_B, 1.0f);
    std::vector<float> h_C(elements_C, 0.0f);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&d_A),
            bytes_A
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&d_B),
            bytes_B
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&d_C),
            bytes_C
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            bytes_A,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            bytes_B,
            cudaMemcpyHostToDevice
        )
    );

    SgemmDispatchInfo dispatch_info =
        select_sgemm_kernel(
            d_A,
            d_B,
            M,
            N,
            K
        );

    CUDA_CHECK(
        launch_sgemm_dispatch(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K,
            nullptr,
            &dispatch_info
        )
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(
            h_C.data(),
            d_C,
            bytes_C,
            cudaMemcpyDeviceToHost
        )
    );

    const float expected_value =
        static_cast<float>(K);

    std::size_t mismatches = 0;
    double max_abs_error = 0.0;

    for (float value : h_C) {
        const double error =
            std::abs(
                static_cast<double>(value) -
                static_cast<double>(expected_value)
            );

        if (error > max_abs_error) {
            max_abs_error = error;
        }

        if (
            !std::isfinite(value) ||
            error > 1e-3
        ) {
            ++mismatches;
        }
    }

    const std::string selected_kernel =
        sgemm_kernel_name(dispatch_info.kernel);

    const bool kernel_matches =
        selected_kernel == expected_kernel;

    std::cout
        << "----------------------------------------\n"
        << "M = " << M << "\n"
        << "N = " << N << "\n"
        << "K = " << K << "\n"
        << "Float4 safe = "
        << std::boolalpha
        << dispatch_info.float4_safe
        << "\n"
        << "Selected kernel = "
        << selected_kernel
        << "\n"
        << "Expected kernel = "
        << expected_kernel
        << "\n"
        << "Kernel matches = "
        << kernel_matches
        << "\n"
        << "Max abs error = "
        << max_abs_error
        << "\n"
        << "Mismatches = "
        << mismatches
        << "\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return kernel_matches && mismatches == 0;
}


int main() {
    std::cout
        << "========================================\n"
        << "Stage 7.9 CMake Package Consumer\n"
        << "========================================\n";

    const bool aligned_passed =
        run_case(
            128,
            132,
            116,
            "float4_double_buffer"
        );

    const bool fallback_passed =
        run_case(
            127,
            131,
            113,
            "scalar_single_buffer"
        );

    const bool all_passed =
        aligned_passed && fallback_passed;

    std::cout
        << "========================================\n"
        << "PACKAGE_CONSUMER_PASS = "
        << std::boolalpha
        << all_passed
        << "\n";

    return all_passed
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
