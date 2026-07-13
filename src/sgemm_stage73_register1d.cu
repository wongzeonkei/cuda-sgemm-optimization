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

constexpr int BM = 128;
constexpr int BN = 32;
constexpr int BK = 16;
constexpr int TM = 8;

constexpr int THREAD_ROWS = BM / TM;
constexpr int THREADS_PER_BLOCK = THREAD_ROWS * BN;

static_assert(BM % TM == 0);
static_assert(THREADS_PER_BLOCK == 512);


// One thread computes TM output values in one output column.
__global__ void sgemm_register_1d_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ float shared_A[BM][BK];
    __shared__ float shared_B[BK][BN];

    const int tid = threadIdx.x;

    const int thread_row_group = tid / BN;
    const int local_col = tid % BN;
    const int local_row_base = thread_row_group * TM;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float accumulators[TM] = {0.0f};

    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        // Load A tile: BM x BK.
        for (
            int index = tid;
            index < BM * BK;
            index += THREADS_PER_BLOCK
        ) {
            const int local_row = index / BK;
            const int local_k = index % BK;

            const int global_row = block_row + local_row;
            const int global_k = tile_k + local_k;

            shared_A[local_row][local_k] =
                global_row < M && global_k < K
                ? A[global_row * K + global_k]
                : 0.0f;
        }

        // Load B tile: BK x BN.
        for (
            int index = tid;
            index < BK * BN;
            index += THREADS_PER_BLOCK
        ) {
            const int local_k = index / BN;
            const int tile_col = index % BN;

            const int global_k = tile_k + local_k;
            const int global_col = block_col + tile_col;

            shared_B[local_k][tile_col] =
                global_k < K && global_col < N
                ? B[global_k * N + global_col]
                : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int local_k = 0; local_k < BK; ++local_k) {
            const float b_value =
                shared_B[local_k][local_col];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                accumulators[i] +=
                    shared_A[local_row_base + i][local_k] *
                    b_value;
            }
        }

        __syncthreads();
    }

    const int global_col = block_col + local_col;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int global_row =
            block_row + local_row_base + i;

        if (global_row < M && global_col < N) {
            C[global_row * N + global_col] =
                accumulators[i];
        }
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
            "Output and reference sizes differ."
        );
    }

    ErrorMetrics metrics;
    double absolute_error_sum = 0.0;

    for (std::size_t i = 0; i < output.size(); ++i) {
        const double actual =
            static_cast<double>(output[i]);

        const double expected =
            static_cast<double>(reference[i]);

        const double absolute_error =
            std::abs(actual - expected);

        const double relative_error =
            absolute_error /
            std::max(std::abs(expected), 1e-8);

        metrics.max_abs_error =
            std::max(
                metrics.max_abs_error,
                absolute_error
            );

        metrics.max_relative_error =
            std::max(
                metrics.max_relative_error,
                relative_error
            );

        absolute_error_sum += absolute_error;

        const double allowed_error =
            absolute_tolerance +
            relative_tolerance * std::abs(expected);

        if (
            !std::isfinite(actual) ||
            absolute_error > allowed_error
        ) {
            ++metrics.mismatch_count;
        }
    }

    metrics.mean_abs_error =
        absolute_error_sum /
        static_cast<double>(output.size());

    return metrics;
}


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


float benchmark_register_kernel(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup,
    int iterations
) {
    const dim3 block(THREADS_PER_BLOCK);

    const dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    for (int i = 0; i < warmup; ++i) {
        sgemm_register_1d_kernel<<<grid, block>>>(
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
        sgemm_register_1d_kernel<<<grid, block>>>(
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

    return total_ms /
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

    return total_ms /
           static_cast<float>(iterations);
}


int main(int argc, char** argv) {
    int M = 512;
    int N = 512;
    int K = 512;
    int warmup = 10;
    int iterations = 50;

    std::string csv_path =
        "results/stage73_register1d.csv";

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
        << std::setprecision(6)
        << "========================================\n"
        << "Stage 7.3 1D Register Tiling\n"
        << "========================================\n"
        << "M = " << M << "\n"
        << "N = " << N << "\n"
        << "K = " << K << "\n"
        << "Block tile = " << BM << " x " << BN << "\n"
        << "K tile = " << BK << "\n"
        << "Thread tile = " << TM << " x 1\n"
        << "Threads per block = "
        << THREADS_PER_BLOCK << "\n"
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
    std::vector<float> h_C_register(elements_C);
    std::vector<float> h_C_cublas(elements_C);

    fill_random(h_A, 42);
    fill_random(h_B, 3407);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C_register = nullptr;
    float* d_C_cublas = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C_register, bytes_C));
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

    const float register_ms =
        benchmark_register_kernel(
            d_A,
            d_B,
            d_C_register,
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
            h_C_register.data(),
            d_C_register,
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

    const ErrorMetrics error =
        compare_results(
            h_C_register,
            h_C_cublas
        );

    const double register_gflops =
        calculate_gflops(
            M,
            N,
            K,
            register_ms
        );

    const double cublas_gflops =
        calculate_gflops(
            M,
            N,
            K,
            cublas_ms
        );

    const double cublas_ratio =
        cublas_gflops > 0.0
        ? register_gflops / cublas_gflops * 100.0
        : 0.0;

    std::cout
        << "Register-tiled CUDA\n"
        << "  Mean latency: "
        << register_ms << " ms\n"
        << "  Performance:  "
        << register_gflops << " GFLOPS\n\n"
        << "cuBLAS Pedantic FP32\n"
        << "  Mean latency: "
        << cublas_ms << " ms\n"
        << "  Performance:  "
        << cublas_gflops << " GFLOPS\n\n"
        << "Register/cuBLAS ratio: "
        << cublas_ratio << "%\n\n";

    std::cout
        << std::scientific
        << std::setprecision(10)
        << "Correctness vs cuBLAS\n"
        << "  Max abs error:  "
        << error.max_abs_error << "\n"
        << "  Mean abs error: "
        << error.mean_abs_error << "\n"
        << "  Max rel error:  "
        << error.max_relative_error << "\n"
        << "  Mismatches:     "
        << error.mismatch_count << "\n\n";

    const std::filesystem::path output_path(csv_path);

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
        << "implementation,M,N,K,BM,BN,BK,TM,"
        << "threads_per_block,warmup,iterations,"
        << "mean_latency_ms,gflops,max_abs_error,"
        << "mean_abs_error,mismatch_count,reference\n";

    csv
        << "register_tiled_1d,"
        << M << ","
        << N << ","
        << K << ","
        << BM << ","
        << BN << ","
        << BK << ","
        << TM << ","
        << THREADS_PER_BLOCK << ","
        << warmup << ","
        << iterations << ","
        << register_ms << ","
        << register_gflops << ","
        << error.max_abs_error << ","
        << error.mean_abs_error << ","
        << error.mismatch_count << ","
        << "cublas\n";

    csv
        << "cublas_pedantic_fp32,"
        << M << ","
        << N << ","
        << K << ","
        << "0,0,0,0,0,"
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
    CUDA_CHECK(cudaFree(d_C_register));
    CUDA_CHECK(cudaFree(d_C_cublas));

    return EXIT_SUCCESS;
}
