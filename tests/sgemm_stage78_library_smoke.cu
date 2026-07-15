#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.h"
#include "sgemm_dispatch.h"


double calculate_gflops(
    int M,
    int N,
    int K,
    double milliseconds
) {
    const double operations =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    return operations /
           (milliseconds / 1000.0) /
           1e9;
}


int main(int argc, char** argv) {
    int M = 512;
    int N = 512;
    int K = 512;
    int iterations = 50;

    std::string alignment_mode = "aligned";

    if (argc >= 4) {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    }

    if (argc >= 5) {
        iterations = std::stoi(argv[4]);
    }

    if (argc >= 6) {
        alignment_mode = argv[5];
    }

    if (
        M <= 0 ||
        N <= 0 ||
        K <= 0 ||
        iterations <= 0
    ) {
        std::cerr << "Invalid dimensions or iteration count.\n";
        return EXIT_FAILURE;
    }

    if (
        alignment_mode != "aligned" &&
        alignment_mode != "misalign-a" &&
        alignment_mode != "misalign-b"
    ) {
        std::cerr
            << "Alignment mode must be:\n"
            << "  aligned\n"
            << "  misalign-a\n"
            << "  misalign-b\n";

        return EXIT_FAILURE;
    }

    const bool misalign_a =
        alignment_mode == "misalign-a";

    const bool misalign_b =
        alignment_mode == "misalign-b";

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

    /*
     * All-one matrices provide a simple exact reference:
     *
     * C[i, j] = sum_k 1 * 1 = K
     */
    std::vector<float> h_A(elements_A, 1.0f);
    std::vector<float> h_B(elements_B, 1.0f);
    std::vector<float> h_C(elements_C, 0.0f);

    float* d_A_base = nullptr;
    float* d_B_base = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_A_base,
            bytes_A + 16
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_B_base,
            bytes_B + 16
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_C,
            bytes_C
        )
    );

    float* d_A =
        d_A_base + (misalign_a ? 1 : 0);

    float* d_B =
        d_B_base + (misalign_b ? 1 : 0);

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

    SgemmDispatchInfo decision =
        select_sgemm_kernel(
            d_A,
            d_B,
            M,
            N,
            K
        );

    constexpr int warmup = 5;

    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(
            launch_sgemm_dispatch(
                d_A,
                d_B,
                d_C,
                M,
                N,
                K
            )
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(
            launch_sgemm_dispatch(
                d_A,
                d_B,
                d_C,
                M,
                N,
                K
            )
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );

    const float mean_ms =
        total_ms /
        static_cast<float>(iterations);

    CUDA_CHECK(
        cudaMemcpy(
            h_C.data(),
            d_C,
            bytes_C,
            cudaMemcpyDeviceToHost
        )
    );

    const float expected =
        static_cast<float>(K);

    double max_abs_error = 0.0;
    std::size_t mismatch_count = 0;

    for (float value : h_C) {
        const double error =
            std::abs(
                static_cast<double>(value) -
                static_cast<double>(expected)
            );

        max_abs_error =
            std::max(max_abs_error, error);

        if (
            !std::isfinite(value) ||
            error > 1e-3
        ) {
            ++mismatch_count;
        }
    }

    const double gflops =
        calculate_gflops(
            M,
            N,
            K,
            mean_ms
        );

    std::cout
        << std::boolalpha
        << std::fixed
        << std::setprecision(6)
        << "========================================\n"
        << "Stage 7.8 Static Library Smoke Test\n"
        << "========================================\n"
        << "M = " << M << "\n"
        << "N = " << N << "\n"
        << "K = " << K << "\n"
        << "Alignment mode = "
        << alignment_mode << "\n"
        << "A address = "
        << static_cast<const void*>(d_A) << "\n"
        << "B address = "
        << static_cast<const void*>(d_B) << "\n"
        << "Dimensions float4 compatible = "
        << decision.dimensions_float4_compatible
        << "\n"
        << "Pointers 16-byte aligned = "
        << decision.pointers_16byte_aligned
        << "\n"
        << "Float4 safe = "
        << decision.float4_safe
        << "\n"
        << "Selected kernel = "
        << sgemm_kernel_name(decision.kernel)
        << "\n"
        << "Mean latency = "
        << mean_ms << " ms\n"
        << "Performance = "
        << gflops << " GFLOPS\n"
        << "Expected value = "
        << expected << "\n"
        << "Max abs error = "
        << max_abs_error << "\n"
        << "Mismatches = "
        << mismatch_count << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A_base));
    CUDA_CHECK(cudaFree(d_B_base));
    CUDA_CHECK(cudaFree(d_C));

    return mismatch_count == 0
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
