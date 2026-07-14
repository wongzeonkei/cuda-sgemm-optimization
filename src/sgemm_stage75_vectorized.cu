#include <algorithm>
#include <cmath>
#include <cstdint>
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
constexpr int BN = 128;
constexpr int BK = 8;

constexpr int TM = 8;
constexpr int TN = 8;

constexpr int VECTOR_WIDTH = 4;

constexpr int THREAD_ROWS = BM / TM;
constexpr int THREAD_COLS = BN / TN;
constexpr int THREADS_PER_BLOCK =
    THREAD_ROWS * THREAD_COLS;

static_assert(BM % TM == 0);
static_assert(BN % TN == 0);
static_assert(BK % VECTOR_WIDTH == 0);
static_assert(BN % VECTOR_WIDTH == 0);
static_assert(THREADS_PER_BLOCK == 256);


/*
 * VECTOR_LOAD = false:
 *     Scalar global-memory loading.
 *
 * VECTOR_LOAD = true:
 *     float4 global-memory loading.
 *
 * The arithmetic and register tiling are identical. This allows
 * the experiment to isolate the effect of vectorized loading.
 */
template<bool VECTOR_LOAD>
__global__ void sgemm_register_2d_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    /*
     * A is transposed when stored in shared memory.
     *
     * Global A tile: [BM, BK]
     * Shared A:      [BK, BM + padding]
     */
    __shared__ float shared_A[BK][BM + 4];
    __shared__ float shared_B[BK][BN];

    const int tid = threadIdx.x;

    const int thread_tile_row =
        tid / THREAD_COLS;

    const int thread_tile_col =
        tid % THREAD_COLS;

    const int local_row_base =
        thread_tile_row * TM;

    const int local_col_base =
        thread_tile_col * TN;

    const int block_row =
        blockIdx.y * BM;

    const int block_col =
        blockIdx.x * BN;

    float accumulators[TM][TN] = {0.0f};

    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        /*
         * A tile:
         *
         * BM * BK = 128 * 8 = 1024 floats.
         * 1024 / 4 = 256 float4 vectors.
         *
         * Each of the 256 threads loads one float4.
         */
        constexpr int A_VECTORS_PER_ROW =
            BK / VECTOR_WIDTH;

        const int a_local_row =
            tid / A_VECTORS_PER_ROW;

        const int a_vector_index =
            tid % A_VECTORS_PER_ROW;

        const int a_local_k =
            a_vector_index * VECTOR_WIDTH;

        const int a_global_row =
            block_row + a_local_row;

        const int a_global_k =
            tile_k + a_local_k;

        if constexpr (VECTOR_LOAD) {
            float4 a_vector =
                make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            /*
             * The vector load is safe only when all four
             * elements are inside the matrix.
             */
            if (
                a_global_row < M &&
                a_global_k + VECTOR_WIDTH - 1 < K
            ) {
                const float* source =
                    A + a_global_row * K + a_global_k;

                a_vector =
                    *reinterpret_cast<const float4*>(source);
            } else {
                /*
                 * Boundary fallback for the final partial K tile
                 * or an out-of-range matrix row.
                 */
                float values[VECTOR_WIDTH] = {
                    0.0f, 0.0f, 0.0f, 0.0f
                };

                #pragma unroll
                for (int lane = 0; lane < VECTOR_WIDTH; ++lane) {
                    const int global_k =
                        a_global_k + lane;

                    if (
                        a_global_row < M &&
                        global_k < K
                    ) {
                        values[lane] =
                            A[a_global_row * K + global_k];
                    }
                }

                a_vector = make_float4(
                    values[0],
                    values[1],
                    values[2],
                    values[3]
                );
            }

            shared_A[a_local_k + 0][a_local_row] =
                a_vector.x;

            shared_A[a_local_k + 1][a_local_row] =
                a_vector.y;

            shared_A[a_local_k + 2][a_local_row] =
                a_vector.z;

            shared_A[a_local_k + 3][a_local_row] =
                a_vector.w;
        } else {
            #pragma unroll
            for (int lane = 0; lane < VECTOR_WIDTH; ++lane) {
                const int global_k =
                    a_global_k + lane;

                shared_A[a_local_k + lane][a_local_row] =
                    a_global_row < M && global_k < K
                    ? A[a_global_row * K + global_k]
                    : 0.0f;
            }
        }

        /*
         * B tile:
         *
         * BK * BN = 8 * 128 = 1024 floats.
         * 1024 / 4 = 256 float4 vectors.
         *
         * Each thread loads one float4.
         */
        constexpr int B_VECTORS_PER_ROW =
            BN / VECTOR_WIDTH;

        const int b_local_k =
            tid / B_VECTORS_PER_ROW;

        const int b_vector_index =
            tid % B_VECTORS_PER_ROW;

        const int b_local_col =
            b_vector_index * VECTOR_WIDTH;

        const int b_global_k =
            tile_k + b_local_k;

        const int b_global_col =
            block_col + b_local_col;

        if constexpr (VECTOR_LOAD) {
            float4 b_vector =
                make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            if (
                b_global_k < K &&
                b_global_col + VECTOR_WIDTH - 1 < N
            ) {
                const float* source =
                    B + b_global_k * N + b_global_col;

                b_vector =
                    *reinterpret_cast<const float4*>(source);
            } else {
                float values[VECTOR_WIDTH] = {
                    0.0f, 0.0f, 0.0f, 0.0f
                };

                #pragma unroll
                for (int lane = 0; lane < VECTOR_WIDTH; ++lane) {
                    const int global_col =
                        b_global_col + lane;

                    if (
                        b_global_k < K &&
                        global_col < N
                    ) {
                        values[lane] =
                            B[b_global_k * N + global_col];
                    }
                }

                b_vector = make_float4(
                    values[0],
                    values[1],
                    values[2],
                    values[3]
                );
            }

            shared_B[b_local_k][b_local_col + 0] =
                b_vector.x;

            shared_B[b_local_k][b_local_col + 1] =
                b_vector.y;

            shared_B[b_local_k][b_local_col + 2] =
                b_vector.z;

            shared_B[b_local_k][b_local_col + 3] =
                b_vector.w;
        } else {
            #pragma unroll
            for (int lane = 0; lane < VECTOR_WIDTH; ++lane) {
                const int global_col =
                    b_global_col + lane;

                shared_B[b_local_k][b_local_col + lane] =
                    b_global_k < K && global_col < N
                    ? B[b_global_k * N + global_col]
                    : 0.0f;
            }
        }

        __syncthreads();

        /*
         * Register-level 8 x 8 outer product.
         */
        #pragma unroll
        for (int local_k = 0; local_k < BK; ++local_k) {
            float register_A[TM];
            float register_B[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                register_A[i] =
                    shared_A[local_k][local_row_base + i];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                register_B[j] =
                    shared_B[local_k][local_col_base + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accumulators[i][j] +=
                        register_A[i] * register_B[j];
                }
            }
        }

        __syncthreads();
    }

    /*
     * Keep scalar stores in this stage so the only main
     * optimization variable is global-memory loading.
     */
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int global_row =
            block_row + local_row_base + i;

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int global_col =
                block_col + local_col_base + j;

            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] =
                    accumulators[i][j];
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
            relative_tolerance *
            std::abs(expected);

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


template<bool VECTOR_LOAD>
float benchmark_kernel(
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
        sgemm_register_2d_kernel<VECTOR_LOAD>
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
        sgemm_register_2d_kernel<VECTOR_LOAD>
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
        "results/stage75_vectorized.csv";

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

    const bool dimensions_support_float4 =
        K % VECTOR_WIDTH == 0 &&
        N % VECTOR_WIDTH == 0;

    cudaFuncAttributes scalar_attributes{};
    cudaFuncAttributes vector_attributes{};

    CUDA_CHECK(
        cudaFuncGetAttributes(
            &scalar_attributes,
            sgemm_register_2d_kernel<false>
        )
    );

    CUDA_CHECK(
        cudaFuncGetAttributes(
            &vector_attributes,
            sgemm_register_2d_kernel<true>
        )
    );

    std::cout
        << std::fixed
        << std::setprecision(6)
        << "========================================\n"
        << "Stage 7.5 float4 Vectorized Loading\n"
        << "========================================\n"
        << "M = " << M << "\n"
        << "N = " << N << "\n"
        << "K = " << K << "\n"
        << "Block tile = " << BM << " x " << BN << "\n"
        << "K tile = " << BK << "\n"
        << "Thread tile = " << TM << " x " << TN << "\n"
        << "Threads per block = "
        << THREADS_PER_BLOCK << "\n"
        << "Vector width = " << VECTOR_WIDTH
        << " floats / 16 bytes\n"
        << "float4 path enabled = "
        << (dimensions_support_float4 ? "yes" : "no")
        << "\n"
        << "Scalar kernel registers = "
        << scalar_attributes.numRegs << "\n"
        << "Vector kernel registers = "
        << vector_attributes.numRegs << "\n"
        << "Vector kernel shared memory = "
        << vector_attributes.sharedSizeBytes
        << " bytes\n"
        << "Vector kernel local memory = "
        << vector_attributes.localSizeBytes
        << " bytes\n"
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
    std::vector<float> h_C_scalar(elements_C);
    std::vector<float> h_C_selected(elements_C);
    std::vector<float> h_C_cublas(elements_C);

    fill_random(h_A, 42);
    fill_random(h_B, 3407);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C_scalar = nullptr;
    float* d_C_selected = nullptr;
    float* d_C_cublas = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C_scalar, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_C_selected, bytes_C));
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

    const bool pointers_aligned =
        reinterpret_cast<std::uintptr_t>(d_A) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(d_B) % 16 == 0;

    const bool use_float4 =
        dimensions_support_float4 &&
        pointers_aligned;

    std::cout
        << "CUDA pointer alignment = "
        << (pointers_aligned ? "16-byte aligned" : "not aligned")
        << "\n"
        << "Selected path = "
        << (use_float4 ? "float4" : "scalar fallback")
        << "\n\n";

    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));

    CUBLAS_CHECK(
        cublasSetMathMode(
            handle,
            CUBLAS_PEDANTIC_MATH
        )
    );

    const float scalar_ms =
        benchmark_kernel<false>(
            d_A,
            d_B,
            d_C_scalar,
            M,
            N,
            K,
            warmup,
            iterations
        );

    float selected_ms = 0.0f;

    if (use_float4) {
        selected_ms =
            benchmark_kernel<true>(
                d_A,
                d_B,
                d_C_selected,
                M,
                N,
                K,
                warmup,
                iterations
            );
    } else {
        selected_ms =
            benchmark_kernel<false>(
                d_A,
                d_B,
                d_C_selected,
                M,
                N,
                K,
                warmup,
                iterations
            );
    }

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
            h_C_scalar.data(),
            d_C_scalar,
            bytes_C,
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            h_C_selected.data(),
            d_C_selected,
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

    const ErrorMetrics scalar_error =
        compare_results(
            h_C_scalar,
            h_C_cublas
        );

    const ErrorMetrics selected_error =
        compare_results(
            h_C_selected,
            h_C_cublas
        );

    const ErrorMetrics selected_vs_scalar =
        compare_results(
            h_C_selected,
            h_C_scalar
        );

    const double scalar_gflops =
        calculate_gflops(
            M,
            N,
            K,
            scalar_ms
        );

    const double selected_gflops =
        calculate_gflops(
            M,
            N,
            K,
            selected_ms
        );

    const double cublas_gflops =
        calculate_gflops(
            M,
            N,
            K,
            cublas_ms
        );

    const double speedup =
        selected_ms > 0.0f
        ? scalar_ms / selected_ms
        : 0.0;

    const double cublas_ratio =
        cublas_gflops > 0.0
        ? selected_gflops / cublas_gflops * 100.0
        : 0.0;

    std::cout
        << "Scalar-load Register2D\n"
        << "  Mean latency: "
        << scalar_ms << " ms\n"
        << "  Performance:  "
        << scalar_gflops << " GFLOPS\n\n"
        << (use_float4
            ? "float4-load Register2D\n"
            : "Scalar fallback Register2D\n")
        << "  Mean latency: "
        << selected_ms << " ms\n"
        << "  Performance:  "
        << selected_gflops << " GFLOPS\n\n"
        << "cuBLAS Pedantic FP32\n"
        << "  Mean latency: "
        << cublas_ms << " ms\n"
        << "  Performance:  "
        << cublas_gflops << " GFLOPS\n\n"
        << "Selected/scalar speedup: "
        << speedup << "x\n"
        << "Selected/cuBLAS ratio:   "
        << cublas_ratio << "%\n\n";

    std::cout
        << std::scientific
        << std::setprecision(10)
        << "Selected path vs cuBLAS\n"
        << "  Max abs error:  "
        << selected_error.max_abs_error << "\n"
        << "  Mean abs error: "
        << selected_error.mean_abs_error << "\n"
        << "  Max rel error:  "
        << selected_error.max_relative_error << "\n"
        << "  Mismatches:     "
        << selected_error.mismatch_count << "\n\n"
        << "Selected path vs scalar kernel\n"
        << "  Max abs error:  "
        << selected_vs_scalar.max_abs_error << "\n"
        << "  Mean abs error: "
        << selected_vs_scalar.mean_abs_error << "\n"
        << "  Mismatches:     "
        << selected_vs_scalar.mismatch_count << "\n\n";

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
        << "implementation,M,N,K,BM,BN,BK,TM,TN,"
        << "vector_width,float4_enabled,"
        << "registers_per_thread,shared_bytes,"
        << "local_memory_bytes,warmup,iterations,"
        << "mean_latency_ms,gflops,max_abs_error,"
        << "mean_abs_error,mismatch_count,reference\n";

    csv
        << "register2d_scalar_load,"
        << M << ","
        << N << ","
        << K << ","
        << BM << ","
        << BN << ","
        << BK << ","
        << TM << ","
        << TN << ","
        << "1,false,"
        << scalar_attributes.numRegs << ","
        << scalar_attributes.sharedSizeBytes << ","
        << scalar_attributes.localSizeBytes << ","
        << warmup << ","
        << iterations << ","
        << scalar_ms << ","
        << scalar_gflops << ","
        << scalar_error.max_abs_error << ","
        << scalar_error.mean_abs_error << ","
        << scalar_error.mismatch_count << ","
        << "cublas\n";

    csv
        << (use_float4
            ? "register2d_float4_load,"
            : "register2d_scalar_fallback,")
        << M << ","
        << N << ","
        << K << ","
        << BM << ","
        << BN << ","
        << BK << ","
        << TM << ","
        << TN << ","
        << VECTOR_WIDTH << ","
        << (use_float4 ? "true," : "false,")
        << (use_float4
            ? vector_attributes.numRegs
            : scalar_attributes.numRegs)
        << ","
        << (use_float4
            ? vector_attributes.sharedSizeBytes
            : scalar_attributes.sharedSizeBytes)
        << ","
        << (use_float4
            ? vector_attributes.localSizeBytes
            : scalar_attributes.localSizeBytes)
        << ","
        << warmup << ","
        << iterations << ","
        << selected_ms << ","
        << selected_gflops << ","
        << selected_error.max_abs_error << ","
        << selected_error.mean_abs_error << ","
        << selected_error.mismatch_count << ","
        << "cublas\n";

    csv
        << "cublas_pedantic_fp32,"
        << M << ","
        << N << ","
        << K << ","
        << "0,0,0,0,0,0,false,0,0,0,"
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
    CUDA_CHECK(cudaFree(d_C_scalar));
    CUDA_CHECK(cudaFree(d_C_selected));
    CUDA_CHECK(cudaFree(d_C_cublas));

    return EXIT_SUCCESS;
}
