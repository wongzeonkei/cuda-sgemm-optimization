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
