#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cuda_check.h"

constexpr int TILE_SIZE = 32;


// ============================================================
// Naive SGEMM
//
// Each CUDA thread computes one output element:
//
// C[row, col] = sum(A[row, k] * B[k, col])
// ============================================================
__global__ void sgemm_naive_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    const int col =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < K; ++k) {
        sum +=
            A[row * K + k] *
            B[k * N + col];
    }

    C[row * N + col] = sum;
}


// ============================================================
// Shared Memory Tiled SGEMM
//
// Block tile: 32 x 32
// Thread block: 32 x 32 = 1024 threads
//
// Each block calculates one 32 x 32 output tile.
// A and B tiles are loaded cooperatively into shared memory.
// ============================================================
template<int TILE>
__global__ void sgemm_shared_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ float shared_A[TILE][TILE];
    __shared__ float shared_B[TILE][TILE];

    const int local_col = threadIdx.x;
    const int local_row = threadIdx.y;

    const int global_col =
        blockIdx.x * TILE + local_col;

    const int global_row =
        blockIdx.y * TILE + local_row;

    float accumulator = 0.0f;

    const int tile_count =
        (K + TILE - 1) / TILE;

    for (int tile = 0; tile < tile_count; ++tile) {
        const int a_col =
            tile * TILE + local_col;

        const int b_row =
            tile * TILE + local_row;

        // Cooperatively load one A tile.
        if (global_row < M && a_col < K) {
            shared_A[local_row][local_col] =
                A[global_row * K + a_col];
        } else {
            shared_A[local_row][local_col] = 0.0f;
        }

        // Cooperatively load one B tile.
        if (b_row < K && global_col < N) {
            shared_B[local_row][local_col] =
                B[b_row * N + global_col];
        } else {
            shared_B[local_row][local_col] = 0.0f;
        }

        // Ensure the whole tile has been loaded.
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            accumulator +=
                shared_A[local_row][k] *
                shared_B[k][local_col];
        }

        // Ensure all threads have finished using the tile
        // before the shared-memory arrays are overwritten.
        __syncthreads();
    }

    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] =
            accumulator;
    }
}


struct ErrorMetrics {
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
    double max_relative_error = 0.0;
    std::size_t mismatch_count = 0;
};


void fill_random(
    std::vector<float>& values,
    unsigned int seed
) {
    std::mt19937 generator(seed);

    std::uniform_real_distribution<float> distribution(
        -1.0f,
        1.0f
    );

    for (float& value : values) {
        value = distribution(generator);
    }
}


ErrorMetrics compare_results(
    const std::vector<float>& output,
    const std::vector<float>& reference,
    float absolute_tolerance = 1e-2f,
    float relative_tolerance = 1e-3f
) {
    if (output.size() != reference.size()) {
        throw std::runtime_error(
            "Output and reference sizes do not match."
        );
    }

    ErrorMetrics metrics;
    double sum_abs_error = 0.0;

    for (std::size_t i = 0; i < output.size(); ++i) {
        const double output_value =
            static_cast<double>(output[i]);

        const double reference_value =
            static_cast<double>(reference[i]);

        const double abs_error =
            std::abs(output_value - reference_value);

        const double denominator =
            std::max(std::abs(reference_value), 1e-8);

        const double relative_error =
            abs_error / denominator;

        metrics.max_abs_error =
            std::max(
                metrics.max_abs_error,
                abs_error
            );

        metrics.max_relative_error =
            std::max(
                metrics.max_relative_error,
                relative_error
            );

        sum_abs_error += abs_error;

        const double allowed_error =
            absolute_tolerance +
            relative_tolerance *
            std::abs(reference_value);

        if (abs_error > allowed_error) {
            ++metrics.mismatch_count;
        }
    }

    metrics.mean_abs_error =
        sum_abs_error /
        static_cast<double>(output.size());

    return metrics;
}


double calculate_gflops(
    int M,
    int N,
    int K,
    double milliseconds
) {
    const double floating_point_operations =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    const double seconds =
        milliseconds / 1000.0;

    return
        floating_point_operations /
        seconds /
        1e9;
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

    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return
        total_ms /
        static_cast<float>(iterations);
}


float benchmark_tiled(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup,
    int iterations
) {
    const dim3 block(
        TILE_SIZE,
        TILE_SIZE
    );

    const dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    for (int i = 0; i < warmup; ++i) {
        sgemm_shared_tiled_kernel<TILE_SIZE>
            <<<grid, block>>>(
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
        sgemm_shared_tiled_kernel<TILE_SIZE>
            <<<grid, block>>>(
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

    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return
        total_ms /
        static_cast<float>(iterations);
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

    auto run = [&]() {
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
        run();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        run();
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

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return
        total_ms /
        static_cast<float>(iterations);
}


int main(int argc, char** argv) {
    int M = 512;
    int N = 512;
    int K = 512;
    int warmup = 10;
    int iterations = 50;

    std::string csv_path =
        "results/stage72_tiled.csv";

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
        std::cerr << "Invalid arguments.\n";
        return EXIT_FAILURE;
    }

    std::cout
        << std::fixed
        << std::setprecision(6);

    std::cout
        << "========================================\n"
        << "Stage 7.2 Shared Memory Tiling\n"
        << "========================================\n"
        << "M = " << M << "\n"
        << "N = " << N << "\n"
        << "K = " << K << "\n"
        << "Tile size = " << TILE_SIZE << " x "
        << TILE_SIZE << "\n"
        << "Threads per block = "
        << TILE_SIZE * TILE_SIZE << "\n"
        << "Warmup = " << warmup << "\n"
        << "Iterations = " << iterations << "\n\n";

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
    std::vector<float> h_C_tiled(elements_C);
    std::vector<float> h_C_cublas(elements_C);

    fill_random(h_A, 42);
    fill_random(h_B, 3407);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C_naive = nullptr;
    float* d_C_tiled = nullptr;
    float* d_C_cublas = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));

    CUDA_CHECK(cudaMalloc(&d_C_naive, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_tiled, bytes_C));
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

    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));

    CUBLAS_CHECK(
        cublasSetMathMode(
            handle,
            CUBLAS_PEDANTIC_MATH
        )
    );

    const float naive_ms =
        benchmark_naive(
            d_A,
            d_B,
            d_C_naive,
            M,
            N,
            K,
            warmup,
            iterations
        );

    const float tiled_ms =
        benchmark_tiled(
            d_A,
            d_B,
            d_C_tiled,
            M,
            N,
            K,
            warmup,
            iterations
        );

    const float cublas_ms =
        benchmark_cublas(
            handle,
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
            h_C_tiled.data(),
            d_C_tiled,
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

    const ErrorMetrics naive_error =
        compare_results(
            h_C_naive,
            h_C_cublas
        );

    const ErrorMetrics tiled_error =
        compare_results(
            h_C_tiled,
            h_C_cublas
        );

    const double naive_gflops =
        calculate_gflops(
            M,
            N,
            K,
            naive_ms
        );

    const double tiled_gflops =
        calculate_gflops(
            M,
            N,
            K,
            tiled_ms
        );

    const double cublas_gflops =
        calculate_gflops(
            M,
            N,
            K,
            cublas_ms
        );

    const double tiled_speedup =
        tiled_ms > 0.0f
        ? naive_ms / tiled_ms
        : 0.0;

    const double tiled_cublas_ratio =
        cublas_gflops > 0.0
        ? tiled_gflops / cublas_gflops * 100.0
        : 0.0;

    std::cout
        << "Naive CUDA\n"
        << "  Mean latency: "
        << naive_ms << " ms\n"
        << "  Performance:  "
        << naive_gflops << " GFLOPS\n\n";

    std::cout
        << "Shared Tiled CUDA\n"
        << "  Mean latency: "
        << tiled_ms << " ms\n"
        << "  Performance:  "
        << tiled_gflops << " GFLOPS\n\n";

    std::cout
        << "cuBLAS Pedantic FP32\n"
        << "  Mean latency: "
        << cublas_ms << " ms\n"
        << "  Performance:  "
        << cublas_gflops << " GFLOPS\n\n";

    std::cout
        << "Performance comparison\n"
        << "  Tiled speedup over naive: "
        << tiled_speedup << "x\n"
        << "  Tiled/cuBLAS ratio:       "
        << tiled_cublas_ratio << "%\n\n";

    std::cout
        << std::scientific
        << std::setprecision(10);

    std::cout
        << "Naive vs cuBLAS correctness\n"
        << "  Max abs error:  "
        << naive_error.max_abs_error << "\n"
        << "  Mean abs error: "
        << naive_error.mean_abs_error << "\n"
        << "  Max rel error:  "
        << naive_error.max_relative_error << "\n"
        << "  Mismatches:     "
        << naive_error.mismatch_count << "\n\n";

    std::cout
        << "Tiled vs cuBLAS correctness\n"
        << "  Max abs error:  "
        << tiled_error.max_abs_error << "\n"
        << "  Mean abs error: "
        << tiled_error.mean_abs_error << "\n"
        << "  Max rel error:  "
        << tiled_error.max_relative_error << "\n"
        << "  Mismatches:     "
        << tiled_error.mismatch_count << "\n\n";

    const std::filesystem::path output_path(
        csv_path
    );

    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(
            output_path.parent_path()
        );
    }

    std::ofstream csv(csv_path);

    if (!csv) {
        throw std::runtime_error(
            "Failed to create CSV: " + csv_path
        );
    }

    csv
        << "implementation,M,N,K,tile_size,"
        << "warmup,iterations,mean_latency_ms,"
        << "gflops,max_abs_error,mean_abs_error,"
        << "mismatch_count,reference\n";

    csv
        << "naive_cuda,"
        << M << ","
        << N << ","
        << K << ","
        << "0,"
        << warmup << ","
        << iterations << ","
        << naive_ms << ","
        << naive_gflops << ","
        << naive_error.max_abs_error << ","
        << naive_error.mean_abs_error << ","
        << naive_error.mismatch_count << ","
        << "cublas\n";

    csv
        << "shared_tiled_cuda,"
        << M << ","
        << N << ","
        << K << ","
        << TILE_SIZE << ","
        << warmup << ","
        << iterations << ","
        << tiled_ms << ","
        << tiled_gflops << ","
        << tiled_error.max_abs_error << ","
        << tiled_error.mean_abs_error << ","
        << tiled_error.mismatch_count << ","
        << "cublas\n";

    csv
        << "cublas_pedantic_fp32,"
        << M << ","
        << N << ","
        << K << ","
        << "0,"
        << warmup << ","
        << iterations << ","
        << cublas_ms << ","
        << cublas_gflops << ","
        << "0,0,0,none\n";

    std::cout
        << std::fixed
        << std::setprecision(6)
        << "Saved CSV: "
        << csv_path << "\n";

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_naive));
    CUDA_CHECK(cudaFree(d_C_tiled));
    CUDA_CHECK(cudaFree(d_C_cublas));

    return EXIT_SUCCESS;
}
