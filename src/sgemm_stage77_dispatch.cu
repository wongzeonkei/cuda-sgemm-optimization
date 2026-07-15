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

constexpr int A_SHARED_STRIDE = BM + 4;
constexpr int A_TILE_ELEMENTS = BK * A_SHARED_STRIDE;
constexpr int B_TILE_ELEMENTS = BK * BN;

static_assert(BM % TM == 0);
static_assert(BN % TN == 0);
static_assert(BK % VECTOR_WIDTH == 0);
static_assert(BN % VECTOR_WIDTH == 0);
static_assert(THREADS_PER_BLOCK == 256);


// ============================================================
// Load four consecutive A values.
//
// Each thread loads one float4:
// 256 threads x 4 floats = 1024 floats = BM x BK.
// ============================================================
template<bool VECTOR_LOAD>
__device__ __forceinline__
float4 load_a_vector(
    const float* __restrict__ A,
    int M,
    int K,
    int block_row,
    int tile_k,
    int tid
) {
    constexpr int vectors_per_row =
        BK / VECTOR_WIDTH;

    const int local_row =
        tid / vectors_per_row;

    const int vector_index =
        tid % vectors_per_row;

    const int local_k =
        vector_index * VECTOR_WIDTH;

    const int global_row =
        block_row + local_row;

    const int global_k =
        tile_k + local_k;

    if constexpr (VECTOR_LOAD) {
        if (
            global_row < M &&
            global_k + VECTOR_WIDTH - 1 < K
        ) {
            const float* source =
                A + global_row * K + global_k;

            return
                *reinterpret_cast<const float4*>(source);
        }
    }

    float values[VECTOR_WIDTH] = {
        0.0f,
        0.0f,
        0.0f,
        0.0f
    };

    #pragma unroll
    for (int lane = 0; lane < VECTOR_WIDTH; ++lane) {
        const int current_k =
            global_k + lane;

        if (
            global_row < M &&
            current_k < K
        ) {
            values[lane] =
                A[global_row * K + current_k];
        }
    }

    return make_float4(
        values[0],
        values[1],
        values[2],
        values[3]
    );
}


// ============================================================
// Load four consecutive B values.
//
// Each thread loads one float4:
// 256 threads x 4 floats = 1024 floats = BK x BN.
// ============================================================
template<bool VECTOR_LOAD>
__device__ __forceinline__
float4 load_b_vector(
    const float* __restrict__ B,
    int K,
    int N,
    int block_col,
    int tile_k,
    int tid
) {
    constexpr int vectors_per_row =
        BN / VECTOR_WIDTH;

    const int local_k =
        tid / vectors_per_row;

    const int vector_index =
        tid % vectors_per_row;

    const int local_col =
        vector_index * VECTOR_WIDTH;

    const int global_k =
        tile_k + local_k;

    const int global_col =
        block_col + local_col;

    if constexpr (VECTOR_LOAD) {
        if (
            global_k < K &&
            global_col + VECTOR_WIDTH - 1 < N
        ) {
            const float* source =
                B + global_k * N + global_col;

            return
                *reinterpret_cast<const float4*>(source);
        }
    }

    float values[VECTOR_WIDTH] = {
        0.0f,
        0.0f,
        0.0f,
        0.0f
    };

    #pragma unroll
    for (int lane = 0; lane < VECTOR_WIDTH; ++lane) {
        const int current_col =
            global_col + lane;

        if (
            global_k < K &&
            current_col < N
        ) {
            values[lane] =
                B[global_k * N + current_col];
        }
    }

    return make_float4(
        values[0],
        values[1],
        values[2],
        values[3]
    );
}


// ============================================================
// Store prefetched float4 values into one shared-memory buffer.
//
// A is transposed while being stored:
// global A [BM, BK] -> shared A [BK, BM + padding]
// ============================================================
__device__ __forceinline__
void store_tile_vectors(
    float* shared_A,
    float* shared_B,
    int buffer_index,
    int tid,
    float4 a_vector,
    float4 b_vector
) {
    constexpr int a_vectors_per_row =
        BK / VECTOR_WIDTH;

    const int a_local_row =
        tid / a_vectors_per_row;

    const int a_vector_index =
        tid % a_vectors_per_row;

    const int a_local_k =
        a_vector_index * VECTOR_WIDTH;

    const int a_buffer_offset =
        buffer_index * A_TILE_ELEMENTS;

    shared_A[
        a_buffer_offset +
        (a_local_k + 0) * A_SHARED_STRIDE +
        a_local_row
    ] = a_vector.x;

    shared_A[
        a_buffer_offset +
        (a_local_k + 1) * A_SHARED_STRIDE +
        a_local_row
    ] = a_vector.y;

    shared_A[
        a_buffer_offset +
        (a_local_k + 2) * A_SHARED_STRIDE +
        a_local_row
    ] = a_vector.z;

    shared_A[
        a_buffer_offset +
        (a_local_k + 3) * A_SHARED_STRIDE +
        a_local_row
    ] = a_vector.w;

    constexpr int b_vectors_per_row =
        BN / VECTOR_WIDTH;

    const int b_local_k =
        tid / b_vectors_per_row;

    const int b_vector_index =
        tid % b_vectors_per_row;

    const int b_local_col =
        b_vector_index * VECTOR_WIDTH;

    const int b_buffer_offset =
        buffer_index * B_TILE_ELEMENTS;

    shared_B[
        b_buffer_offset +
        b_local_k * BN +
        b_local_col + 0
    ] = b_vector.x;

    shared_B[
        b_buffer_offset +
        b_local_k * BN +
        b_local_col + 1
    ] = b_vector.y;

    shared_B[
        b_buffer_offset +
        b_local_k * BN +
        b_local_col + 2
    ] = b_vector.z;

    shared_B[
        b_buffer_offset +
        b_local_k * BN +
        b_local_col + 3
    ] = b_vector.w;
}


// ============================================================
// Calculate one 8 x 8 thread tile from one shared-memory K tile.
// ============================================================
__device__ __forceinline__
void compute_shared_tile(
    const float* shared_A,
    const float* shared_B,
    int buffer_index,
    int local_row_base,
    int local_col_base,
    float* accumulators
) {
    const int a_buffer_offset =
        buffer_index * A_TILE_ELEMENTS;

    const int b_buffer_offset =
        buffer_index * B_TILE_ELEMENTS;

    #pragma unroll
    for (int local_k = 0; local_k < BK; ++local_k) {
        float register_A[TM];
        float register_B[TN];

        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            register_A[i] =
                shared_A[
                    a_buffer_offset +
                    local_k * A_SHARED_STRIDE +
                    local_row_base + i
                ];
        }

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            register_B[j] =
                shared_B[
                    b_buffer_offset +
                    local_k * BN +
                    local_col_base + j
                ];
        }

        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                accumulators[i * TN + j] +=
                    register_A[i] *
                    register_B[j];
            }
        }
    }
}


// ============================================================
// Single-buffer baseline.
//
// This is the Stage 7.5 execution structure.
// It is included in the same executable for a fair comparison.
// ============================================================
template<bool VECTOR_LOAD>
__global__ void sgemm_single_buffer_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ float shared_A[BK][A_SHARED_STRIDE];
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

    float accumulators[TM * TN] = {0.0f};

    for (int tile_k = 0; tile_k < K; tile_k += BK) {
        const float4 a_vector =
            load_a_vector<VECTOR_LOAD>(
                A,
                M,
                K,
                block_row,
                tile_k,
                tid
            );

        const float4 b_vector =
            load_b_vector<VECTOR_LOAD>(
                B,
                K,
                N,
                block_col,
                tile_k,
                tid
            );

        store_tile_vectors(
            &shared_A[0][0],
            &shared_B[0][0],
            0,
            tid,
            a_vector,
            b_vector
        );

        __syncthreads();

        compute_shared_tile(
            &shared_A[0][0],
            &shared_B[0][0],
            0,
            local_row_base,
            local_col_base,
            accumulators
        );

        __syncthreads();
    }

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
                    accumulators[i * TN + j];
            }
        }
    }
}


// ============================================================
// Double-buffered SGEMM.
//
// Pipeline:
// 1. Load tile 0 into shared buffer 0.
// 2. Prefetch tile t+1 from global memory into registers.
// 3. Compute tile t from the current shared buffer.
// 4. Store prefetched registers into the other shared buffer.
// 5. Synchronize once and switch buffers.
//
// The global loads for tile t+1 are independent of the arithmetic
// using tile t, allowing part of the memory latency to be hidden.
// ============================================================
template<bool VECTOR_LOAD>
__global__ void sgemm_double_buffer_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ float shared_A[2][BK][A_SHARED_STRIDE];
    __shared__ float shared_B[2][BK][BN];

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

    const int tile_count =
        (K + BK - 1) / BK;

    float accumulators[TM * TN] = {0.0f};

    /*
     * Prologue: load the first K tile into buffer 0.
     */
    const float4 first_a =
        load_a_vector<VECTOR_LOAD>(
            A,
            M,
            K,
            block_row,
            0,
            tid
        );

    const float4 first_b =
        load_b_vector<VECTOR_LOAD>(
            B,
            K,
            N,
            block_col,
            0,
            tid
        );

    store_tile_vectors(
        &shared_A[0][0][0],
        &shared_B[0][0][0],
        0,
        tid,
        first_a,
        first_b
    );

    __syncthreads();

    /*
     * Main software pipeline.
     */
    for (
        int tile_index = 0;
        tile_index < tile_count;
        ++tile_index
    ) {
        const int current_buffer =
            tile_index & 1;

        const int next_buffer =
            current_buffer ^ 1;

        const bool has_next_tile =
            tile_index + 1 < tile_count;

        float4 next_a =
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        float4 next_b =
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        /*
         * Issue global-memory loads for the next tile.
         * Their results remain in registers while the current
         * shared-memory tile is calculated.
         */
        if (has_next_tile) {
            const int next_tile_k =
                (tile_index + 1) * BK;

            next_a =
                load_a_vector<VECTOR_LOAD>(
                    A,
                    M,
                    K,
                    block_row,
                    next_tile_k,
                    tid
                );

            next_b =
                load_b_vector<VECTOR_LOAD>(
                    B,
                    K,
                    N,
                    block_col,
                    next_tile_k,
                    tid
                );
        }

        /*
         * Calculate the current tile while next_a and next_b
         * wait in registers.
         */
        compute_shared_tile(
            &shared_A[0][0][0],
            &shared_B[0][0][0],
            current_buffer,
            local_row_base,
            local_col_base,
            accumulators
        );

        /*
         * Write the prefetched tile into the inactive buffer.
         */
        if (has_next_tile) {
            store_tile_vectors(
                &shared_A[0][0][0],
                &shared_B[0][0][0],
                next_buffer,
                tid,
                next_a,
                next_b
            );
        }

        /*
         * One barrier:
         * - all threads finished reading the current buffer;
         * - all threads finished writing the next buffer.
         */
        __syncthreads();
    }

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
                    accumulators[i * TN + j];
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


template<bool VECTOR_LOAD, bool DOUBLE_BUFFER>
void launch_kernel(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K
) {
    const dim3 block(THREADS_PER_BLOCK);

    const dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    if constexpr (DOUBLE_BUFFER) {
        sgemm_double_buffer_kernel<VECTOR_LOAD>
            <<<grid, block>>>(
                d_A,
                d_B,
                d_C,
                M,
                N,
                K
            );
    } else {
        sgemm_single_buffer_kernel<VECTOR_LOAD>
            <<<grid, block>>>(
                d_A,
                d_B,
                d_C,
                M,
                N,
                K
            );
    }
}


template<bool VECTOR_LOAD, bool DOUBLE_BUFFER>
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
    for (int i = 0; i < warmup; ++i) {
        launch_kernel<VECTOR_LOAD, DOUBLE_BUFFER>(
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
        launch_kernel<VECTOR_LOAD, DOUBLE_BUFFER>(
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



#include "sgemm_dispatch.h"


const char* sgemm_kernel_name(
    SgemmKernelKind kernel
) {
    switch (kernel) {
        case SgemmKernelKind::ScalarSingleBuffer:
            return "scalar_single_buffer";

        case SgemmKernelKind::Float4DoubleBuffer:
            return "float4_double_buffer";
    }

    return "unknown";
}


SgemmDispatchInfo select_sgemm_kernel(
    const float* d_A,
    const float* d_B,
    int M,
    int N,
    int K
) {
    const bool valid_dimensions =
        M > 0 &&
        N > 0 &&
        K > 0;

    const bool dimensions_float4_compatible =
        valid_dimensions &&
        N % VECTOR_WIDTH == 0 &&
        K % VECTOR_WIDTH == 0;

    const bool pointers_16byte_aligned =
        d_A != nullptr &&
        d_B != nullptr &&
        reinterpret_cast<std::uintptr_t>(d_A) % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(d_B) % 16 == 0;

    const bool float4_safe =
        dimensions_float4_compatible &&
        pointers_16byte_aligned;

    return {
        float4_safe
            ? SgemmKernelKind::Float4DoubleBuffer
            : SgemmKernelKind::ScalarSingleBuffer,
        valid_dimensions,
        dimensions_float4_compatible,
        pointers_16byte_aligned,
        float4_safe
    };
}


cudaError_t launch_sgemm_dispatch(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    cudaStream_t stream,
    SgemmDispatchInfo* dispatch_info
) {
    if (
        d_A == nullptr ||
        d_B == nullptr ||
        d_C == nullptr ||
        M <= 0 ||
        N <= 0 ||
        K <= 0
    ) {
        return cudaErrorInvalidValue;
    }

    const SgemmDispatchInfo decision =
        select_sgemm_kernel(
            d_A,
            d_B,
            M,
            N,
            K
        );

    if (dispatch_info != nullptr) {
        *dispatch_info = decision;
    }

    const dim3 block(THREADS_PER_BLOCK);

    const dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    switch (decision.kernel) {
        case SgemmKernelKind::Float4DoubleBuffer:
            sgemm_double_buffer_kernel<true>
                <<<grid, block, 0, stream>>>(
                    d_A,
                    d_B,
                    d_C,
                    M,
                    N,
                    K
                );
            break;

        case SgemmKernelKind::ScalarSingleBuffer:
            sgemm_single_buffer_kernel<false>
                <<<grid, block, 0, stream>>>(
                    d_A,
                    d_B,
                    d_C,
                    M,
                    N,
                    K
                );
            break;
    }

    return cudaGetLastError();
}


float benchmark_dispatch(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int N,
    int K,
    int warmup,
    int iterations,
    SgemmDispatchInfo* dispatch_info
) {
    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(
            launch_sgemm_dispatch(
                d_A,
                d_B,
                d_C,
                M,
                N,
                K,
                nullptr,
                dispatch_info
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
                K,
                nullptr,
                dispatch_info
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

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms /
           static_cast<float>(iterations);
}


int main(int argc, char** argv) {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int warmup = 10;
    int iterations = 50;

    std::string csv_path =
        "results/stage77_dispatch.csv";

    std::string alignment_mode =
        "aligned";

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

    if (argc >= 8) {
        alignment_mode = argv[7];
    }

    if (
        M <= 0 ||
        N <= 0 ||
        K <= 0 ||
        warmup < 0 ||
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
            << "Alignment mode must be one of:\n"
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

    std::vector<float> h_A(elements_A);
    std::vector<float> h_B(elements_B);
    std::vector<float> h_C_dispatch(elements_C);
    std::vector<float> h_C_cublas(elements_C);

    fill_random(h_A, 42);
    fill_random(h_B, 3407);

    float* d_A_base = nullptr;
    float* d_B_base = nullptr;

    float* d_C_dispatch = nullptr;
    float* d_C_cublas = nullptr;

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
            &d_C_dispatch,
            bytes_C
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_C_cublas,
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

    const SgemmDispatchInfo initial_decision =
        select_sgemm_kernel(
            d_A,
            d_B,
            M,
            N,
            K
        );

    cudaFuncAttributes kernel_attributes{};

    switch (initial_decision.kernel) {
        case SgemmKernelKind::Float4DoubleBuffer:
            CUDA_CHECK(
                cudaFuncGetAttributes(
                    &kernel_attributes,
                    sgemm_double_buffer_kernel<true>
                )
            );
            break;

        case SgemmKernelKind::ScalarSingleBuffer:
            CUDA_CHECK(
                cudaFuncGetAttributes(
                    &kernel_attributes,
                    sgemm_single_buffer_kernel<false>
                )
            );
            break;
    }

    std::cout
        << std::fixed
        << std::setprecision(6)
        << "========================================\n"
        << "Stage 7.7 SGEMM Kernel Dispatcher\n"
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
        << std::boolalpha
        << initial_decision.dimensions_float4_compatible
        << "\n"
        << "Pointers 16-byte aligned = "
        << initial_decision.pointers_16byte_aligned
        << "\n"
        << "Float4 safe = "
        << initial_decision.float4_safe
        << "\n"
        << "Selected kernel = "
        << sgemm_kernel_name(initial_decision.kernel)
        << "\n"
        << "Registers per thread = "
        << kernel_attributes.numRegs << "\n"
        << "Shared memory = "
        << kernel_attributes.sharedSizeBytes
        << " bytes\n"
        << "Local memory = "
        << kernel_attributes.localSizeBytes
        << " bytes\n"
        << "Warmup = " << warmup << "\n"
        << "Iterations = " << iterations
        << "\n\n";

    SgemmDispatchInfo measured_decision{};

    const float dispatch_ms =
        benchmark_dispatch(
            d_A,
            d_B,
            d_C_dispatch,
            M,
            N,
            K,
            warmup,
            iterations,
            &measured_decision
        );

    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));

    CUBLAS_CHECK(
        cublasSetMathMode(
            handle,
            CUBLAS_PEDANTIC_MATH
        )
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
            h_C_dispatch.data(),
            d_C_dispatch,
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
            h_C_dispatch,
            h_C_cublas
        );

    const double dispatch_gflops =
        calculate_gflops(
            M,
            N,
            K,
            dispatch_ms
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
        ? dispatch_gflops /
          cublas_gflops *
          100.0
        : 0.0;

    std::cout
        << std::fixed
        << std::setprecision(6)
        << "Dispatched SGEMM\n"
        << "  Mean latency: "
        << dispatch_ms << " ms\n"
        << "  Performance:  "
        << dispatch_gflops << " GFLOPS\n\n"
        << "cuBLAS Pedantic FP32\n"
        << "  Mean latency: "
        << cublas_ms << " ms\n"
        << "  Performance:  "
        << cublas_gflops << " GFLOPS\n\n"
        << "Dispatch/cuBLAS ratio: "
        << cublas_ratio << "%\n\n";

    std::cout
        << std::scientific
        << std::setprecision(10)
        << "Correctness\n"
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
        << "implementation,M,N,K,alignment_mode,"
        << "dimensions_float4_compatible,"
        << "pointers_16byte_aligned,float4_safe,"
        << "registers_per_thread,shared_bytes,"
        << "local_memory_bytes,warmup,iterations,"
        << "mean_latency_ms,gflops,max_abs_error,"
        << "mean_abs_error,mismatch_count,"
        << "cublas_gflops,cublas_ratio_percent\n";

    csv
        << sgemm_kernel_name(
            measured_decision.kernel
        )
        << ","
        << M << ","
        << N << ","
        << K << ","
        << alignment_mode << ","
        << std::boolalpha
        << measured_decision.dimensions_float4_compatible
        << ","
        << measured_decision.pointers_16byte_aligned
        << ","
        << measured_decision.float4_safe
        << ","
        << kernel_attributes.numRegs << ","
        << kernel_attributes.sharedSizeBytes << ","
        << kernel_attributes.localSizeBytes << ","
        << warmup << ","
        << iterations << ","
        << dispatch_ms << ","
        << dispatch_gflops << ","
        << error.max_abs_error << ","
        << error.mean_abs_error << ","
        << error.mismatch_count << ","
        << cublas_gflops << ","
        << cublas_ratio << "\n";

    std::cout
        << std::fixed
        << std::setprecision(6)
        << "Saved CSV: "
        << csv_path << "\n";

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A_base));
    CUDA_CHECK(cudaFree(d_B_base));
    CUDA_CHECK(cudaFree(d_C_dispatch));
    CUDA_CHECK(cudaFree(d_C_cublas));

    return error.mismatch_count == 0
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
