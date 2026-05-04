/*
 * Blocked Cholesky decomposition using:
 *   1. cuSOLVER POTRF  - factorize diagonal block
 *   2. cuBLAS   TRSM   - solve panel
 *   3. cuBLAS   GEMM   - update Schur complement (in place of SYRK)
 *
 * Compile:
 *   nvcc -O3 -arch=sm_90 -std=c++17 -o cholesky cholesky_blocked.cu -lcublas -lcusolver
 *
 * Usage:
 *   ./cholesky [n] [nb] [num_trials] [use_emulation]
 *   ./cholesky 4096 256 10 1          # with FP64 fixed-point emulation
 *   ./cholesky 4096 256 10 0          # native FP64
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t _e = (err);                                                \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(err)                                                      \
    do {                                                                       \
        cublasStatus_t _e = (err);                                             \
        if (_e != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                       \
                    __FILE__, __LINE__, (int)_e);                              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUSOLVER_CHECK(err)                                                    \
    do {                                                                       \
        cusolverStatus_t _e = (err);                                           \
        if (_e != CUSOLVER_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuSOLVER error %s:%d: %d\n",                     \
                    __FILE__, __LINE__, (int)_e);                              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Fixed-point emulation workspace sizing
// (from cuBLAS 13.1 documentation, section 1.5.2.4)
// ---------------------------------------------------------------------------

static inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

size_t getFixedPointWorkspaceSizeInBytes(int m, int n, int k, int batchCount,
                                         bool isComplex,
                                         cudaEmulationMantissaControl mantissaControl,
                                         int maxMantissaBitCount)
{
    constexpr double MULTIPLIER = 1.25;

    int mult       = isComplex ? 2 : 1;
    int numSlices  = ceildiv(maxMantissaBitCount + 1, 8);

    int padded_m     = ceildiv(m, 1024) * 1024;
    int padded_n     = ceildiv(n, 1024) * 1024;
    int padded_k     = ceildiv(k, 128)  * 128;
    int num_blocks_k = ceildiv(k, 64);

    size_t gemm_workspace = sizeof(int8_t) *
        ((size_t)padded_m * padded_k + (size_t)padded_n * padded_k) * mult * numSlices;

    gemm_workspace += sizeof(int32_t) * ((size_t)padded_m + padded_n) * mult;
    if (isComplex) {
        gemm_workspace += sizeof(double) * (size_t)m * n * mult * mult;
    }

    size_t adp_workspace = 0;
    if (mantissaControl == CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC) {
        adp_workspace = sizeof(int32_t) *
            ((size_t)m * num_blocks_k + (size_t)n * num_blocks_k +
             (size_t)m * n) * mult;
    }

    constexpr size_t CONSTANT_SIZE = 128 * 1024 * 1024;
    return (size_t)(std::max(gemm_workspace, adp_workspace) * batchCount * MULTIPLIER)
           + CONSTANT_SIZE;
}

// ---------------------------------------------------------------------------
// Blocked Cholesky (lower triangular, column-major)
//
// On entry:  d_A holds a symmetric positive definite matrix (n x n)
// On exit:   lower triangle of d_A holds L where A = L * L^T
//
// The cuBLAS emulation workspace is set on cublas_handle before this
// function is called and is consumed automatically by DGEMM — no extra
// arguments are needed here.
// ---------------------------------------------------------------------------
void cholesky_blocked(cublasHandle_t     cublas_handle,
                      cusolverDnHandle_t cusolver_handle,
                      double *d_A, int n, int nb,
                      int *d_info, double *d_potrf_work, int lwork)
{
    const double neg_one = -1.0;
    const double one     =  1.0;

    for (int k = 0; k < n; k += nb) {
        int kb = (k + nb < n) ? nb : (n - k);

        double *d_Akk = d_A + (size_t)k * n + k;

        // ----------------------------------------------------------------
        // 1. cuSOLVER POTRF — factorise the diagonal block in-place
        // ----------------------------------------------------------------
        CUSOLVER_CHECK(cusolverDnDpotrf(
            cusolver_handle,
            CUBLAS_FILL_MODE_LOWER,
            kb,
            d_Akk, n,
            d_potrf_work, lwork,
            d_info
        ));

        int m_rem = n - k - kb;
        if (m_rem <= 0) break;

        double *d_panel = d_A + (size_t)k * n + (k + kb);
        double *d_trail = d_A + (size_t)(k + kb) * n + (k + kb);

        // ----------------------------------------------------------------
        // 2. cuBLAS TRSM — solve the panel
        //    panel = panel * L_kk^{-T}
        // ----------------------------------------------------------------
        CUBLAS_CHECK(cublasDtrsm(
            cublas_handle,
            CUBLAS_SIDE_RIGHT,
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T,
            CUBLAS_DIAG_NON_UNIT,
            m_rem, kb,
            &one,
            d_Akk,   n,
            d_panel, n
        ));

        // ----------------------------------------------------------------
        // 3. cuBLAS GEMM — Schur complement update (replaces SYRK)
        //    A_trail -= panel * panel^T
        //
        //    When fixed-point emulation is enabled on the handle, cuBLAS
        //    uses the pre-allocated workspace attached via cublasSetWorkspace.
        //    No change to this call is required.
        // ----------------------------------------------------------------
        CUBLAS_CHECK(cublasDgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            m_rem, m_rem, kb,
            &neg_one,
            d_panel, n,
            d_panel, n,
            &one,
            d_trail, n
        ));
    }
}

// ---------------------------------------------------------------------------
// Helper: generate a symmetric positive-definite matrix on the host
// ---------------------------------------------------------------------------
void generate_spd(double *A, int n)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            A[(size_t)j * n + i] = (double)rand() / RAND_MAX;

    for (int j = 0; j < n; j++) {
        for (int i = j + 1; i < n; i++) {
            double avg = 0.5 * (A[(size_t)j * n + i] + A[(size_t)i * n + j]);
            A[(size_t)j * n + i] = avg;
            A[(size_t)i * n + j] = avg;
        }
        A[(size_t)j * n + j] += (double)n;
    }
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------
void benchmark_cholesky(int n, int nb, int num_warmup, int num_trials,
                         bool use_emulation)
{
    printf("=============================================================\n");
    printf("Blocked Cholesky benchmark\n");
    printf("  Matrix size  : %d x %d (%.1f MB)\n",
           n, n, (double)n * n * sizeof(double) / (1 << 20));
    printf("  Block size   : %d\n", nb);
    printf("  Warmup runs  : %d\n", num_warmup);
    printf("  Timed runs   : %d\n", num_trials);
    printf("  FP emulation : %s\n",
           use_emulation ? "ON (FP64 fixed-point)" : "OFF (native FP64)");
    printf("=============================================================\n");

    // ---- host matrix ----
    size_t bytes = (size_t)n * n * sizeof(double);
    double *h_A  = (double*)malloc(bytes);
    if (!h_A) { fprintf(stderr, "malloc failed\n"); exit(1); }
    generate_spd(h_A, n);

    // ---- device allocations ----
    double *d_A, *d_potrf_work;
    int    *d_info;
    CUDA_CHECK(cudaMalloc(&d_A,    bytes));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // ---- library handles ----
    cublasHandle_t     cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

    // ---- POTRF workspace ----
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        cusolver_handle, CUBLAS_FILL_MODE_LOWER, nb, d_A, n, &lwork));
    CUDA_CHECK(cudaMalloc(&d_potrf_work, (size_t)lwork * sizeof(double)));

    // ---- Fixed-point emulation workspace for DGEMM ----
    //
    // The largest GEMM in the loop is at the first iteration (k=0):
    //   C[(n-nb) x (n-nb)] -= A[(n-nb) x nb] * B^T[nb x (n-nb)]
    // i.e. m = n-nb, n_dim = n-nb, k = nb.
    //
    // We size the workspace for this call and allocate once before the loop.
    // All subsequent (smaller) GEMM calls reuse the same buffer — it is
    // always large enough because workspace size grows with problem size.
    //
    // ADP (dynamic mantissa control) with 63 bits is the cuBLAS default
    // and guarantees equal or better accuracy than native FP64.
    //
    void  *d_emul_workspace     = nullptr;
    size_t emul_workspace_bytes = 0;

    if (use_emulation) {
        int largest_m = n - nb;
        int largest_n = n - nb;
        int largest_k = nb;

        emul_workspace_bytes = getFixedPointWorkspaceSizeInBytes(
            largest_m, largest_n, largest_k,
            /*batchCount=*/      1,
            /*isComplex=*/       false,
            CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC,
            /*maxMantissaBitCount=*/ 63
        );

        printf("  Emulation WS : %.2f MB\n",
               (double)emul_workspace_bytes / (1 << 20));

        // Allocate — cudaMalloc guarantees >= 256-byte alignment, satisfying
        // the cublasSetWorkspace alignment requirement.
        CUDA_CHECK(cudaMalloc(&d_emul_workspace, emul_workspace_bytes));

        // Bind to handle. Every subsequent cuBLAS call on this handle
        // (including DGEMM) will use this buffer automatically.
        CUBLAS_CHECK(cublasSetWorkspace(
            cublas_handle, d_emul_workspace, emul_workspace_bytes));

        // Enable FP64 fixed-point emulation math mode.
        // Requires CUDA Toolkit >= 13.0 Update 2.
        // If your toolkit is older, you can instead set the environment
        // variable CUBLAS_EMULATE_DOUBLE_PRECISION=1 before launching.
        CUBLAS_CHECK(cublasSetMathMode(
            cublas_handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
    }

    // ---- CUDA events for timing ----
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // ---- warmup ----
    for (int i = 0; i < num_warmup; i++) {
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        cholesky_blocked(cublas_handle, cusolver_handle,
                         d_A, n, nb, d_info, d_potrf_work, lwork);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- timed runs ----
    float total_ms = 0.0f;
    for (int i = 0; i < num_trials; i++) {
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(ev_start));

        cholesky_blocked(cublas_handle, cusolver_handle,
                         d_A, n, nb, d_info, d_potrf_work, lwork);

        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        total_ms += ms;
        printf("  Trial %2d: %.3f ms\n", i + 1, ms);
    }

    float  avg_ms = total_ms / num_trials;
    double flops  = (double)n * n * n / 3.0;
    double gflops = flops / (avg_ms * 1e-3) / 1e9;

    printf("-------------------------------------------------------------\n");
    printf("  Average time  : %.3f ms\n", avg_ms);
    printf("  GFLOP/s       : %.2f\n",    gflops);
    printf("=============================================================\n");

    // ---- cleanup ----
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    free(h_A);
    cudaFree(d_A);
    cudaFree(d_info);
    cudaFree(d_potrf_work);
    if (d_emul_workspace) cudaFree(d_emul_workspace);
    cublasDestroy(cublas_handle);
    cusolverDnDestroy(cusolver_handle);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    srand(42);

    int  n             = 4096;
    int  nb            = 256;
    int  num_warmup    = 3;
    int  num_trials    = 10;
    bool use_emulation = false;

    if (argc > 1) n             = atoi(argv[1]);
    if (argc > 2) nb            = atoi(argv[2]);
    if (argc > 3) num_trials    = atoi(argv[3]);
    if (argc > 4) use_emulation = (atoi(argv[4]) != 0);

    benchmark_cholesky(n, nb, num_warmup, num_trials, use_emulation);
    return 0;
}
