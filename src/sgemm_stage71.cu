#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda_check.h"


// Row-major:
// A: [M, K]
// B: [K, N]
// C: [M, N]
__global__ void sgemm_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}


void fill_random(std::vector<float>& data, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (float& value : data) {
        value = distribution(generator);
    }
}


void cpu_sgemm(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M,
    int N,
    int K
) {
    std::fill(C.begin(), C.end(), 0.0f);

    // i-k-j loop order gives better cache behavior than i-j-k.
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const float a = A[i * K + k];

            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}


struct ErrorMetrics {
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double max_relative_error = 0.0;
    std::size_t mismatch_count = 0;
};


ErrorMetrics compare_results(
    const std::vector<float>& output,
    const std::vector<float>& reference,
    float absolute_tolerance = 1e-2f,
    float relative_tolerance = 1e-3f
) {
    if (output.size() != reference.size()) {
        throw std::runtime_error("Output and reference sizes do not match.");
    }

    ErrorMetrics metrics;
    double absolute_error_sum = 0.0;

    for (std::size_t i = 0; i < output.size(); ++i) {
        const double output_value = static_cast<double>(output[i]);
        const double reference_value = static_cast<double>(reference[i]);

        const double absolute_error =
            std::abs(output_value - reference_value);

        const double denominator =
            std::max(std::abs(reference_value), 1e-8);

        const double relative_error =
            absolute_error / denominator;

        metrics.max_abs_error =
            std::max(metrics.max_abs_error, absolute_error);

        metrics.max_relative_error =
            std::max(metrics.max_relative_error, relative_error);

        absolute_error_sum += absolute_error;

        const double allowed_error =
            absolute_tolerance +
            relative_tolerance * std::abs(reference_value);

        if (absolute_error > allowed_error) {
            ++metrics.mismatch_count;
        }
    }

    metrics.mean_abs_error =
        absolute_error_sum / static_cast<double>(output.size());

    return metrics;
}


double compute_gflops(int M, int N, int K, double milliseconds) {
    const double operations =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    const double seconds = milliseconds / 1000.0;

    return operations / seconds / 1e9;
}


float benchmark_naive(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup,
    int iterations
) {
    const dim3 block(16, 16);
    const dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    for (int i = 0; i < warmup; ++i) {
        sgemm_naive_kernel<<<grid, block>>>(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        sgemm_naive_kernel<<<grid, block>>>(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / static_cast<float>(iterations);
}


float benchmark_cublas(
    cublasHandle_t handle,
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup,
    int iterations
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    /*
     * Host matrices use row-major layout.
     *
     * cuBLAS interprets memory as column-major. Therefore:
     *
     * C_row = A_row * B_row
     *
     * is evaluated as:
     *
     * C_col^T = B_col^T * A_col^T
     *
     * using dimensions N x M x K.
     */
    auto run_cublas = [&]() {
        CUBLAS_CHECK(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                d_B,
                N,
                d_A,
                K,
                &beta,
                d_C,
                N
            )
        );
    };

    for (int i = 0; i < warmup; ++i) {
        run_cublas();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        run_cublas();
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / static_cast<float>(iterations);
}


int main(int argc, char** argv) {
    int M = 512;
    int N = 512;
    int K = 512;
    int warmup = 10;
    int iterations = 50;
    std::string csv_path = "results/stage71_baseline.csv";

    if (argc >= 4) {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    }

    if (argc >= 5) {
        warmup = std::stoi(argv[4]);
    }

    if (argc >= 6) {
        iterations = std::stoi(argv[5]);
    }

    if (argc >= 7) {
        csv_path = argv[6];
    }

    if (
        M <= 0 ||
        N <= 0 ||
        K <= 0 ||
        warmup < 0 ||
        iterations <= 0
    ) {
        std::cerr << "Invalid benchmark arguments.\n";
        return EXIT_FAILURE;
    }

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "========================================\n";
    std::cout << "Stage 7.1 SGEMM Baseline\n";
    std::cout << "========================================\n";
    std::cout << "Matrix layout: row-major\n";
    std::cout << "M = " << M << "\n";
    std::cout << "N = " << N << "\n";
    std::cout << "K = " << K << "\n";
    std::cout << "Warmup = " << warmup << "\n";
    std::cout << "Iterations = " << iterations << "\n\n";

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

    std::vector<float> h_A(elements_A);
    std::vector<float> h_B(elements_B);
    std::vector<float> h_C_naive(elements_C);
    std::vector<float> h_C_cublas(elements_C);
    std::vector<float> h_C_cpu(elements_C);

    fill_random(h_A, 42);
    fill_random(h_B, 3407);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C_naive = nullptr;
    float* d_C_cublas = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C_naive, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, bytes_C));

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

    cublasHandle_t cublas_handle{};
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Prevent an implicit TF32 comparison on Ampere.
    CUBLAS_CHECK(
        cublasSetMathMode(
            cublas_handle,
            CUBLAS_PEDANTIC_MATH
        )
    );

    const float naive_ms = benchmark_naive(
        d_A,
        d_B,
        d_C_naive,
        M,
        N,
        K,
        warmup,
        iterations
    );

    const float cublas_ms = benchmark_cublas(
        cublas_handle,
        d_A,
        d_B,
        d_C_cublas,
        M,
        N,
        K,
        warmup,
        iterations
    );

    CUDA_CHECK(
        cudaMemcpy(
            h_C_naive.data(),
            d_C_naive,
            bytes_C,
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            h_C_cublas.data(),
            d_C_cublas,
            bytes_C,
            cudaMemcpyDeviceToHost
        )
    );

    const ErrorMetrics naive_vs_cublas =
        compare_results(h_C_naive, h_C_cublas);

    std::cout << "Naive CUDA\n";
    std::cout << "  Mean latency: "
              << naive_ms << " ms\n";
    std::cout << "  Performance:  "
              << compute_gflops(M, N, K, naive_ms)
              << " GFLOPS\n\n";

    std::cout << "cuBLAS SGEMM (pedantic FP32)\n";
    std::cout << "  Mean latency: "
              << cublas_ms << " ms\n";
    std::cout << "  Performance:  "
              << compute_gflops(M, N, K, cublas_ms)
              << " GFLOPS\n\n";

    std::cout << "Naive CUDA vs cuBLAS\n";
    std::cout << "  Max abs error:  "
              << naive_vs_cublas.max_abs_error << "\n";
    std::cout << "  Mean abs error: "
              << naive_vs_cublas.mean_abs_error << "\n";
    std::cout << "  Max rel error:  "
              << naive_vs_cublas.max_relative_error << "\n";
    std::cout << "  Mismatches:     "
              << naive_vs_cublas.mismatch_count << "\n\n";

    bool cpu_reference_ran = false;
    ErrorMetrics naive_vs_cpu{};
    ErrorMetrics cublas_vs_cpu{};

    const std::uint64_t cpu_operation_limit =
        static_cast<std::uint64_t>(512) *
        512 *
        512;

    const std::uint64_t problem_size =
        static_cast<std::uint64_t>(M) *
        N *
        K;

    if (problem_size <= cpu_operation_limit) {
        std::cout << "Running CPU reference...\n";

        cpu_sgemm(
            h_A,
            h_B,
            h_C_cpu,
            M,
            N,
            K
        );

        cpu_reference_ran = true;

        naive_vs_cpu =
            compare_results(h_C_naive, h_C_cpu);

        cublas_vs_cpu =
            compare_results(h_C_cublas, h_C_cpu);

        std::cout << "Naive CUDA vs CPU\n";
        std::cout << "  Max abs error:  "
                  << naive_vs_cpu.max_abs_error << "\n";
        std::cout << "  Mean abs error: "
                  << naive_vs_cpu.mean_abs_error << "\n";
        std::cout << "  Mismatches:     "
                  << naive_vs_cpu.mismatch_count << "\n\n";

        std::cout << "cuBLAS vs CPU\n";
        std::cout << "  Max abs error:  "
                  << cublas_vs_cpu.max_abs_error << "\n";
        std::cout << "  Mean abs error: "
                  << cublas_vs_cpu.mean_abs_error << "\n";
        std::cout << "  Mismatches:     "
                  << cublas_vs_cpu.mismatch_count << "\n\n";
    } else {
        std::cout
            << "CPU reference skipped because the problem is larger "
            << "than 512^3 operations.\n\n";
    }

    const double naive_gflops =
        compute_gflops(M, N, K, naive_ms);

    const double cublas_gflops =
        compute_gflops(M, N, K, cublas_ms);

    const double cublas_ratio =
        cublas_gflops > 0.0
        ? naive_gflops / cublas_gflops * 100.0
        : 0.0;

    std::cout << "Naive/cuBLAS performance ratio: "
              << cublas_ratio << "%\n";

    const std::filesystem::path csv_file(csv_path);

    if (csv_file.has_parent_path()) {
        std::filesystem::create_directories(
            csv_file.parent_path()
        );
    }

    std::ofstream csv(csv_path);

    if (!csv) {
        throw std::runtime_error(
            "Failed to open CSV output: " + csv_path
        );
    }

    csv << "implementation,M,N,K,warmup,iterations,"
           "mean_latency_ms,gflops,max_abs_error,"
           "mean_abs_error,mismatch_count,reference\n";

    csv << "naive_cuda,"
        << M << ","
        << N << ","
        << K << ","
        << warmup << ","
        << iterations << ","
        << naive_ms << ","
        << naive_gflops << ","
        << naive_vs_cublas.max_abs_error << ","
        << naive_vs_cublas.mean_abs_error << ","
        << naive_vs_cublas.mismatch_count << ","
        << "cublas\n";

    csv << "cublas_pedantic_fp32,"
        << M << ","
        << N << ","
        << K << ","
        << warmup << ","
        << iterations << ","
        << cublas_ms << ","
        << cublas_gflops << ","
        << (cpu_reference_ran
            ? cublas_vs_cpu.max_abs_error
            : 0.0)
        << ","
        << (cpu_reference_ran
            ? cublas_vs_cpu.mean_abs_error
            : 0.0)
        << ","
        << (cpu_reference_ran
            ? cublas_vs_cpu.mismatch_count
            : 0)
        << ","
        << (cpu_reference_ran ? "cpu" : "none")
        << "\n";

    std::cout << "Saved CSV: " << csv_path << "\n";

    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_naive));
    CUDA_CHECK(cudaFree(d_C_cublas));

    return EXIT_SUCCESS;
}
