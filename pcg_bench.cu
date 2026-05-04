/*
 * pcg_benchmark.cu
 *
 * Preconditioned Conjugate Gradient in FP64, comparing different
 * single-precision Cholesky factorizations as preconditioners.
 *
 * Preconditioners:
 *   0) None (plain CG)
 *   1) cuSOLVER Spotrf (FP32)
 *   2) Mixed FP16 Cholesky  (cuBLASLt FP16 GEMM, FP32 accum/output)
 *   3) Mixed FP8 Cholesky   (cuBLASLt FP8 tensorwise GEMM, FP32 accum/output)
 *   4) Mixed FP8 Cholesky   (cuBLASLt FP8 block-scaled GEMM, FP32 accum/output)
 *   5) Adaptive-precision Hybrid Cholesky (per-step precision selection)
 *
 * CG runs in FP32.  Each preconditioner L is computed in FP32 (or mixed).
 * Preconditioner application:  z = L^{-T}(L^{-1} r)
 * via two cuBLAS FP32 triangular solves.
 *
 * Test matrix:  A = Q diag(λ) Q^T   with eigenvalues from 1 to 1/κ
 * under various distributions.  Built as FP64 then cast to FP32.
 *
 * Build (sm_90a, CUTLASS 3.7+):
 *   nvcc -std=c++17 -O3 -arch=sm_90a -Xptxas --allow-expensive-optimizations=true --expt-relaxed-constexpr -I../cutlass/include  -I../cutlass/tools/util/include  pcg_bench.cu mmio.c  -lcusolver -lcublas -lcurand -o pcg_bench
 *
 * Run:
 *   ./pcg_bench --n 4096 --nb 256 --cond 1e6
 *   ./pcg_bench --n 4096 --nb 256 --cond 1e8 --csv results.csv
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cfloat>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <curand.h>


#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/util/packed_stride.hpp>

extern "C" {
#include "mmio.h"
}

using namespace cute;

#define QUANT_TILE 32

#define COUNT_UNDERFLOW

// ─── Small kernel: x[i] -= 1.0  (for forward-error vs exact solution = ones) ──
__global__ void subtract_ones_f64_kernel(double* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] -= 1.0;
}

// ─── Small kernel: random normal init for Lanczos starting vector ───────────
__global__ void init_lanczos_start_kernel(double* __restrict__ v, int n, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // simple hash-based PRNG — good enough for Lanczos start vectors
    unsigned h = (unsigned)i * 2654435761u + seed;
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
    // map to uniform in [-1, 1)
    v[i] = ((double)h / 2147483648.0) - 1.0;
}

// ─── NaN detection on a double vector (returns host int) ─────────────────────
__global__ void vec_has_nan_f64_kernel(const double* __restrict__ v, size_t n, int* flag) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && isnan(v[i])) atomicExch(flag, 1);
}

static inline int vec_has_nan_f64(const double* d_v, size_t n) {
    int* d_flag; int h_flag = 0;
    cudaMalloc(&d_flag, sizeof(int)); cudaMemset(d_flag, 0, sizeof(int));
    int bl = 256;
    size_t gr = (n + bl - 1) / bl;
    vec_has_nan_f64_kernel<<<(unsigned)gr, bl>>>(d_v, n, d_flag);
    cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_flag);
    return h_flag;
}


static inline double compute_forward_error_from_ones(
        const double* d_x, int n, cublasHandle_t blas)
{
    double* d_err;
    cudaMalloc(&d_err, sizeof(double) * n);
    cudaMemcpy(d_err, d_x, sizeof(double) * n, cudaMemcpyDeviceToDevice);

    int bl = 256, gr = (n + bl - 1) / bl;
    subtract_ones_f64_kernel<<<gr, bl>>>(d_err, n);

    cublasPointerMode_t old_mode;
    cublasGetPointerMode(blas, &old_mode);
    cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST);

    double norm_err = 0.0;
    cublasDnrm2(blas, n, d_err, 1, &norm_err);

    cublasSetPointerMode(blas, old_mode);
    cudaFree(d_err);

    return norm_err / std::sqrt((double)n);
}


static inline double compute_true_residual(
        const double* d_A, const double* d_b, const double* d_x,
        int n, cublasHandle_t blas)
{
    double* d_r;
    cudaMalloc(&d_r, sizeof(double) * n);
    // r = b
    cudaMemcpy(d_r, d_b, sizeof(double) * n, cudaMemcpyDeviceToDevice);

    cublasPointerMode_t old_mode;
    cublasGetPointerMode(blas, &old_mode);
    cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST);

    // r = b - A*x
    double neg = -1.0, one = 1.0;
    cublasDgemv(blas, CUBLAS_OP_N, n, n, &neg, d_A, n, d_x, 1, &one, d_r, 1);

    double norm_r = 0.0, norm_b = 0.0;
    cublasDnrm2(blas, n, d_r, 1, &norm_r);
    cublasDnrm2(blas, n, d_b, 1, &norm_b);

    cublasSetPointerMode(blas, old_mode);
    cudaFree(d_r);

    return (norm_b > 0) ? norm_r / norm_b : norm_r;
}


static inline void lanczos_extreme_eigs(
        const double* d_A, const double* d_L, int n, int k_max,
        double& lambda_min, double& lambda_max,
        cublasHandle_t blas, cusolverDnHandle_t cusolver)
{
    if (k_max > n) k_max = n;

    // Device vectors — basis V (n × k_max), work vectors
    double *d_V, *d_w, *d_tmp;
    cudaMalloc(&d_V,  sizeof(double) * (size_t)n * k_max);
    cudaMalloc(&d_w,  sizeof(double) * n);
    cudaMalloc(&d_tmp,sizeof(double) * n);

    // Host-side tridiag (α on diag, β on off-diag)
    std::vector<double> alpha(k_max, 0.0), beta(k_max, 0.0);

    cublasPointerMode_t old_mode;
    cublasGetPointerMode(blas, &old_mode);
    cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST);

    // v_0 = random, then normalize
    {
        int bl = 256, gr = (n + bl - 1) / bl;
        init_lanczos_start_kernel<<<gr, bl>>>(d_V, n, 0xC0FFEEu);
        double nrm;
        cublasDnrm2(blas, n, d_V, 1, &nrm);
        double inv = 1.0 / nrm;
        cublasDscal(blas, n, &inv, d_V, 1);
    }

    int k_used = k_max;
    for (int j = 0; j < k_max; ++j) {
        double* vj   = d_V + (size_t)j * n;
        double* vjp1 = (j + 1 < k_max) ? d_V + (size_t)(j + 1) * n : d_tmp;

        // w = op(v_j)
        if (d_L == nullptr) {
            // w = A * vj
            double one = 1.0, zero = 0.0;
            cublasDgemv(blas, CUBLAS_OP_N, n, n, &one, d_A, n, vj, 1, &zero, d_w, 1);
        } else {
            // w = L⁻¹ ( A ( L⁻ᵀ vj ) )
            cudaMemcpy(d_w, vj, sizeof(double) * n, cudaMemcpyDeviceToDevice);
            cublasDtrsv(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                        CUBLAS_DIAG_NON_UNIT, n, d_L, n, d_w, 1);
            double one = 1.0, zero = 0.0;
            cublasDgemv(blas, CUBLAS_OP_N, n, n, &one, d_A, n, d_w, 1, &zero, d_tmp, 1);
            cudaMemcpy(d_w, d_tmp, sizeof(double) * n, cudaMemcpyDeviceToDevice);
            cublasDtrsv(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                        CUBLAS_DIAG_NON_UNIT, n, d_L, n, d_w, 1);
        }

        // α_j = v_j^T w
        cublasDdot(blas, n, vj, 1, d_w, 1, &alpha[j]);

        // w -= α_j v_j  (and β_{j-1} v_{j-1} if j > 0)
        double na = -alpha[j];
        cublasDaxpy(blas, n, &na, vj, 1, d_w, 1);
        if (j > 0) {
            double nb = -beta[j - 1];
            double* vjm1 = d_V + (size_t)(j - 1) * n;
            cublasDaxpy(blas, n, &nb, vjm1, 1, d_w, 1);
        }

        // Full reorthogonalization against v_0 … v_j
        for (int i = 0; i <= j; ++i) {
            double* vi = d_V + (size_t)i * n;
            double dot;
            cublasDdot(blas, n, vi, 1, d_w, 1, &dot);
            double ndot = -dot;
            cublasDaxpy(blas, n, &ndot, vi, 1, d_w, 1);
        }

        // β_j = ||w||
        cublasDnrm2(blas, n, d_w, 1, &beta[j]);

        if (beta[j] < 1e-14) { k_used = j + 1; break; }

        if (j + 1 < k_max) {
            double inv = 1.0 / beta[j];
            cudaMemcpy(vjp1, d_w, sizeof(double) * n, cudaMemcpyDeviceToDevice);
            cublasDscal(blas, n, &inv, vjp1, 1);
        }
    }
    cublasSetPointerMode(blas, old_mode);

    // Build tridiagonal as a dense k_used × k_used symmetric matrix on device
    std::vector<double> T_host((size_t)k_used * k_used, 0.0);
    for (int i = 0; i < k_used; ++i) {
        T_host[(size_t)i * k_used + i] = alpha[i];
        if (i + 1 < k_used) {
            T_host[(size_t)i       * k_used + (i + 1)] = beta[i];
            T_host[(size_t)(i + 1) * k_used + i      ] = beta[i];
        }
    }
    double *d_T, *d_evals;
    cudaMalloc(&d_T,     sizeof(double) * (size_t)k_used * k_used);
    cudaMalloc(&d_evals, sizeof(double) * k_used);
    cudaMemcpy(d_T, T_host.data(), sizeof(double) * (size_t)k_used * k_used,
               cudaMemcpyHostToDevice);

    int lwork = 0;
    cusolverDnDsyevd_bufferSize(cusolver,
        CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER,
        k_used, d_T, k_used, d_evals, &lwork);
    double* d_work; int* d_info;
    cudaMalloc(&d_work, sizeof(double) * lwork);
    cudaMalloc(&d_info, sizeof(int));
    cusolverDnDsyevd(cusolver,
        CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER,
        k_used, d_T, k_used, d_evals, d_work, lwork, d_info);

    std::vector<double> evals(k_used);
    cudaMemcpy(evals.data(), d_evals, sizeof(double) * k_used, cudaMemcpyDeviceToHost);

    // Eigenvalues come sorted ascending from Dsyevd
    lambda_min = evals.front();
    lambda_max = evals.back();

    cudaFree(d_T); cudaFree(d_evals); cudaFree(d_work); cudaFree(d_info);
    cudaFree(d_V); cudaFree(d_w); cudaFree(d_tmp);
}

// =============================================================================
//  Convenience wrapper: compute κ = λmax / λmin
// =============================================================================
static inline double estimate_kappa(
        const double* d_A, const double* d_L, int n, int k_max,
        cublasHandle_t blas, cusolverDnHandle_t cusolver)
{
    double lmin, lmax;
    lanczos_extreme_eigs(d_A, d_L, n, k_max, lmin, lmax, blas, cusolver);
    if (lmin <= 0) return -1.0;   // indicator: indefinite or numerical failure
    return lmax / lmin;
}

// =============================================================================
//  ExtraMetrics — aggregates everything for CSV emission.
// =============================================================================
struct ExtraMetrics {
    double forward_err;        // ‖x - ones‖ / sqrt(n)
    double true_resid;         // ‖b - Ax‖ / ‖b‖  (FP64)
    double time_per_iter_ms;   // pcg_ms / iters
    double kappa_A_est;        // Lanczos estimate of κ(A); -1 if not computed
    double kappa_MinvA_est;    // Lanczos estimate of κ(M⁻¹A); -1 if unprec
    int    nan_flag;           // 1 if preconditioner had NaN, else 0
    // Adaptive-only (0 for non-adaptive)
    int    steps_fp8, steps_fp16, steps_tf32, steps_fp32, steps_fp64;
};

static inline ExtraMetrics make_extra_metrics_default() {
    ExtraMetrics m{};
    m.forward_err = -1; m.true_resid = -1; m.time_per_iter_ms = -1;
    m.kappa_A_est = -1; m.kappa_MinvA_est = -1; m.nan_flag = 0;
    m.steps_fp8 = m.steps_fp16 = m.steps_tf32 = m.steps_fp32 = m.steps_fp64 = 0;
    return m;
}




// ─── Eigenvalue distribution enum ────────────────────────────────
enum eigenvalue_dist_t {
    DIST_GEOMETRIC = 1,
    DIST_ARITHMETIC,
    DIST_HALF_HALF,
    DIST_RANDOM,
    DIST_POLY_DECAY,
    DIST_MULTI_CLUSTER,
    DIST_POWER_LAW,
    DIST_COUNT
};

static const eigenvalue_dist_t ALL_DISTS[] = {
    DIST_GEOMETRIC, DIST_ARITHMETIC,
    DIST_HALF_HALF, DIST_RANDOM,
    DIST_POLY_DECAY, DIST_MULTI_CLUSTER, DIST_POWER_LAW
};
static const int NUM_DISTS = sizeof(ALL_DISTS) / sizeof(ALL_DISTS[0]);

const char* dist_name(eigenvalue_dist_t d) {
    switch (d) {
        case DIST_GEOMETRIC:     return "geometric";
        case DIST_ARITHMETIC:    return "arithmetic";
        case DIST_HALF_HALF:     return "half_half";
        case DIST_RANDOM:        return "random";
        case DIST_POLY_DECAY:    return "poly_decay";
        case DIST_MULTI_CLUSTER: return "multi_cluster";
        case DIST_POWER_LAW:     return "power_law";
        default:                 return "unknown";
    }
}

eigenvalue_dist_t dist_from_name(const char* s) {
    if (strcmp(s, "geometric") == 0)          return DIST_GEOMETRIC;
    if (strcmp(s, "arithmetic") == 0)         return DIST_ARITHMETIC;
    if (strcmp(s, "half_half") == 0)          return DIST_HALF_HALF;
    if (strcmp(s, "random") == 0)             return DIST_RANDOM;
    if (strcmp(s, "poly_decay") == 0)         return DIST_POLY_DECAY;
    if (strcmp(s, "multi_cluster") == 0)      return DIST_MULTI_CLUSTER;
    if (strcmp(s, "power_law") == 0)          return DIST_POWER_LAW;
    fprintf(stderr, "Unknown distribution: %s\n", s);
    exit(1);
}


// ─── Error macros ────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                      \
    cudaError_t e = (call);                                        \
    if (e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(e));        \
        exit(1);                                                   \
    }                                                              \
} while (0)

#define CUSOLVER_CHECK(call) do {                                  \
    cusolverStatus_t s = (call);                                   \
    if (s != CUSOLVER_STATUS_SUCCESS) {                            \
        fprintf(stderr, "cuSOLVER %s:%d: %d\n",                   \
                __FILE__, __LINE__, (int)s); exit(1);              \
    }                                                              \
} while (0)

#define CUBLAS_CHECK(call) do {                                    \
    cublasStatus_t s = (call);                                     \
    if (s != CUBLAS_STATUS_SUCCESS) {                              \
        fprintf(stderr, "cuBLAS %s:%d: %d\n",                     \
                __FILE__, __LINE__, (int)s); exit(1);              \
    }                                                              \
} while (0)

#define CURAND_CHECK(call) do {                                    \
    curandStatus_t s = (call);                                     \
    if (s != CURAND_STATUS_SUCCESS) {                              \
        fprintf(stderr, "cuRAND %s:%d: %d\n",                     \
                __FILE__, __LINE__, (int)s); exit(1);              \
    }                                                              \
} while (0)

#define CUTLASS_CHECK(status) do {                                 \
    if ((status) != cutlass::Status::kSuccess) {                   \
        fprintf(stderr, "CUTLASS %s:%d: %d\n",                    \
                __FILE__, __LINE__, (int)(status)); exit(1);       \
    }                                                              \
} while (0)


// ═══════════════════════════════════════════════════════════════════
//  Precision enum for adaptive selection
//
//  Ordered from lowest to highest precision.  The adaptive selector
//  picks the *lowest* precision whose error bound is satisfied.
//
//  TF32 sits between FP16 and FP32: it has the same 10-bit mantissa
//  as FP16 (so eps = 2^-10 ≈ 9.77e-4) but the full FP32 exponent
//  range, and operates directly on FP32 data (no conversion needed).
// ═══════════════════════════════════════════════════════════════════

enum PrecisionChoice {
    PREC_FP8  = 0,
    PREC_FP16 = 1,
    PREC_TF32 = 2,
    PREC_FP32 = 3,
    PREC_FP64 = 4,
    PREC_COUNT = 5
};

static const char* precision_name(PrecisionChoice p) {
    switch (p) {
        case PREC_FP8:  return "FP8";
        case PREC_FP16: return "FP16";
        case PREC_TF32: return "TF32";
        case PREC_FP32: return "FP32";
        case PREC_FP64: return "FP64";
        default:        return "FP64";
    }
}


// ═══════════════════════════════════════════════════════════════════
//  Small utility kernels
// ═══════════════════════════════════════════════════════════════════

__global__ void scale_col_kernel(double* A, int rows, int cols, int col, double s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) A[(size_t)col * rows + i] *= s;
}

__global__ void add_diagonal_f32_kernel(float* A, int n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[(size_t)i * n + i] += val;
}


// ═══════════════════════════════════════════════════════════════════
//  GPU panel Cholesky (spotf2) — single-block right-looking
// ═══════════════════════════════════════════════════════════════════
static constexpr int SPOTRF_PANEL_IB = 32;

// ── Warp-shuffle helpers for double (two 32-bit shuffles) ────────────────
__device__ __forceinline__ double shfl_down_f64(double v, int offset) {
    int lo = __shfl_down_sync(0xffffffff, __double2loint(v), offset);
    int hi = __shfl_down_sync(0xffffffff, __double2hiint(v), offset);
    return __hiloint2double(hi, lo);
}
__device__ __forceinline__ double shfl_xor_f64(double v, int mask) {
    int lo = __shfl_xor_sync(0xffffffff, __double2loint(v), mask);
    int hi = __shfl_xor_sync(0xffffffff, __double2hiint(v), mask);
    return __hiloint2double(hi, lo);
}

// ── Type-dispatched device math ──────────────────────────────────────────
template<typename T> __device__ __forceinline__ T dev_sqrt(T v);
template<> __device__ __forceinline__ float  dev_sqrt(float v)  { return sqrtf(v); }
template<> __device__ __forceinline__ double dev_sqrt(double v) { return sqrt(v); }

template<typename T> __device__ __forceinline__ T dev_fabs(T v);
template<> __device__ __forceinline__ float  dev_fabs(float v)  { return fabsf(v); }
template<> __device__ __forceinline__ double dev_fabs(double v) { return fabs(v); }

template<typename T> __device__ __forceinline__ T dev_fmax(T a, T b);
template<> __device__ __forceinline__ float  dev_fmax(float a, float b)  { return fmaxf(a, b); }
template<> __device__ __forceinline__ double dev_fmax(double a, double b) { return fmax(a, b); }

template<typename T> __device__ __forceinline__ T shfl_down_T(T v, int offset);
template<> __device__ __forceinline__ float  shfl_down_T(float v, int offset) {
    return __shfl_down_sync(0xffffffff, v, offset);
}
template<> __device__ __forceinline__ double shfl_down_T(double v, int offset) {
    return shfl_down_f64(v, offset);
}



__device__ __forceinline__ void print_if_negative(float val, int idx) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (val < 0.0f) {
        printf("Underflow at index %d: %e, for thread %d\n", idx, val, tid);
    }

}

__global__ void coo_to_dense_f64_kernel(
    double* __restrict__ A,
    const int*    __restrict__ row,
    const int*    __restrict__ col,
    const double* __restrict__ val,
    int nnz, int n, int lda, bool symmetric)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;
    int r = row[i];
    int c = col[i];
    A[(size_t)c * lda + r] = val[i];
    if (symmetric && r != c)
        A[(size_t)r * lda + c] = val[i];
}
 
// ── Load a Matrix Market file → dense FP64 matrix on GPU ─────────
//
// Returns 0 on success, nonzero on error.
// Caller must cudaFree(*d_A) when done.
// Only loads real symmetric/general matrices ≤ max_dim.

__global__ void set_diagonal_f64_kernel(double* A, int lda,
                                         int start, int end, double val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i < end) A[(size_t)i * lda + i] = val;
}
 
int load_mtx_to_dense_f64(const char* path, double** d_A, int* out_n,
                           int max_dim = 40000)
{
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return -1;
    }
 
    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "Could not read Matrix Market banner in %s\n", path);
        fclose(f);
        return -1;
    }
 
    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode) || !mm_is_real(matcode)) {
        fprintf(stderr, "Only real coordinate matrices supported (got %s)\n", path);
        fclose(f);
        return -1;
    }
 
    bool symmetric = mm_is_symmetric(matcode);
    bool is_general = mm_is_general(matcode);
 
    int M, N, nnz_file;
    if (mm_read_mtx_crd_size(f, &M, &N, &nnz_file) != 0) {
        fprintf(stderr, "Could not read matrix size from %s\n", path);
        fclose(f);
        return -1;
    }
 
    if (M != N) {
        fprintf(stderr, "Matrix is not square: %d x %d\n", M, N);
        fclose(f);
        return -1;
    }
 
    if (M > max_dim) {
        fprintf(stderr, "Matrix too large: %d > %d max\n", M, max_dim);
        fclose(f);
        return -1;
    }
 
    printf("  Loading %s: %d x %d, %d entries, %s\n",
           path, M, N, nnz_file,
           symmetric ? "symmetric" : "general");
 
    // Read COO triplets on host
    int*    h_row = (int*)malloc(sizeof(int)    * nnz_file);
    int*    h_col = (int*)malloc(sizeof(int)    * nnz_file);
    double* h_val = (double*)malloc(sizeof(double) * nnz_file);
 
    for (int i = 0; i < nnz_file; i++) {
        int r, c;
        double v;
        if (fscanf(f, "%d %d %lg", &r, &c, &v) != 3) {
            fprintf(stderr, "Parse error at entry %d\n", i);
            free(h_row); free(h_col); free(h_val);
            fclose(f);
            return -1;
        }
        h_row[i] = r - 1;  // 1-based → 0-based
        h_col[i] = c - 1;
        h_val[i] = v;
    }
    fclose(f);
 
    int n = M;
    size_t dense_bytes = sizeof(double) * (size_t)n * n;
 
    // Check GPU memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // Need ~3x dense for A + L + workspace
    if (dense_bytes * 3 > free_mem) {
        fprintf(stderr, "Insufficient GPU memory: need ~%.1f GB, have %.1f GB free\n",
                (double)(dense_bytes * 3) / (1024.0*1024.0*1024.0),
                (double)free_mem / (1024.0*1024.0*1024.0));
        free(h_row); free(h_col); free(h_val);
        return -1;
    }
 
    // Allocate dense matrix on GPU and zero it
    double* dA;
    CUDA_CHECK(cudaMalloc(&dA, dense_bytes));
    CUDA_CHECK(cudaMemset(dA, 0, dense_bytes));
 
    // Upload COO to device
    int*    d_row;
    int*    d_col;
    double* d_val;
    CUDA_CHECK(cudaMalloc(&d_row, sizeof(int)    * nnz_file));
    CUDA_CHECK(cudaMalloc(&d_col, sizeof(int)    * nnz_file));
    CUDA_CHECK(cudaMalloc(&d_val, sizeof(double) * nnz_file));
    CUDA_CHECK(cudaMemcpy(d_row, h_row, sizeof(int)    * nnz_file, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, h_col, sizeof(int)    * nnz_file, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, h_val, sizeof(double) * nnz_file, cudaMemcpyHostToDevice));
 
    free(h_row); free(h_col); free(h_val);
 
    // Scatter COO → dense
    int bl = 256, gr = (nnz_file + bl - 1) / bl;
    coo_to_dense_f64_kernel<<<gr, bl>>>(dA, d_row, d_col, d_val,
                                         nnz_file, n, n, symmetric);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
 
    CUDA_CHECK(cudaFree(d_row));
    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_val));
 
    *d_A   = dA;
    *out_n = n;
 
    printf("  Dense matrix: %d x %d (%.2f GB)\n",
           n, n, (double)dense_bytes / (1024.0*1024.0*1024.0));
    // ── Pad to next multiple of nb for CUTLASS alignment ─────────
    //    (identity block on padded diagonal keeps SPD)
    int pad_to = 256;  // must match --nb; or pass nb as a parameter
    int n_padded = ((n + pad_to - 1) / pad_to) * pad_to;

    if (n_padded != n) {
        printf("  Padding %d -> %d (multiple of %d)\n", n, n_padded, pad_to);

        size_t padded_bytes = sizeof(double) * (size_t)n_padded * n_padded;
        double* dA_padded;
        CUDA_CHECK(cudaMalloc(&dA_padded, padded_bytes));
        CUDA_CHECK(cudaMemset(dA_padded, 0, padded_bytes));

        // Copy n×n block into top-left of padded matrix (col-major)
        CUDA_CHECK(cudaMemcpy2D(
            dA_padded,  sizeof(double) * n_padded,  // dst ptr, dst pitch
            dA,         sizeof(double) * n,          // src ptr, src pitch
            sizeof(double) * n, n,                   // width bytes, height
            cudaMemcpyDeviceToDevice));

        // Padded rows/cols get identity diagonal → stays SPD
        int n_extra = n_padded - n;
        int bl = 256, gr = (n_extra + bl - 1) / bl;
        set_diagonal_f64_kernel<<<gr, bl>>>(dA_padded, n_padded, n, n_padded, 1.0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(dA));
        dA = dA_padded;
        n  = n_padded;
    }
 
    return 0;
}
template <typename T>
__global__ void potf2_panel_kernel(
    T* __restrict__ A,
    int n_actual,
    int lda,
    int* __restrict__ info,
    T beta,
    T delta)
{
    constexpr int NB = SPOTRF_PANEL_IB;  // 32

    extern __shared__ char smem_raw[];
    T* sA          = reinterpret_cast<T*>(smem_raw);
    T* dot_scratch = sA + NB * (NB + 1);

    const int tid     = threadIdx.x;
    const int lane    = tid & (warpSize - 1);
    const int warp_id = tid / warpSize;
    const int nwarps  = (blockDim.x + warpSize - 1) / warpSize;

    if (tid == 0) *info = 0;

    // ── 1. Zero tile, load real panel, pad diagonal ──────────────────────
    for (int idx = tid; idx < NB * NB; idx += blockDim.x)
        sA[idx] = T(0);
    __syncthreads();

    for (int idx = tid; idx < n_actual * n_actual; idx += blockDim.x) {
        int col = idx / n_actual;
        int row = idx % n_actual;
        sA[col * NB + row] = A[(size_t)col * lda + row];
    }
    if (tid == 0) {
        for (int j = n_actual; j < NB; j++)
            sA[j * NB + j] = T(1);
    }
    __syncthreads();

    const T b2inv = T(1) / (beta * beta);

    // ── 2. Factorize the full 32×32 tile ─────────────────────────────────
    #pragma unroll
    for (int j = 0; j < NB; j++) {

        T dot = T(0);
        for (int k = tid; k < j; k += blockDim.x) {
            T v = sA[k * NB + j];
            dot += v * v;
        }
        for (int s = warpSize / 2; s > 0; s >>= 1)
            dot += shfl_down_T<T>(dot, s);
        if (lane == 0) dot_scratch[warp_id] = dot;
        __syncthreads();

        T ajj;
        if (tid == 0) {
            T sum = T(0);
            for (int w = 0; w < nwarps; w++) sum += dot_scratch[w];
            ajj = sA[j * NB + j] - sum;
            dot_scratch[0] = ajj;
        }
        __syncthreads();
        ajj = dot_scratch[0];

        if (ajj <= T(0)) {
            if (tid == 0) {
                // Row-∞-norm of the already-computed row-j entries in L
                // (columns 0..j-1 of row j).  Bounds the off-diagonal growth
                // the next trailing update must be dominated by.
                T rinf = T(0);
                for (int k = 0; k < j; k++) {
                    T v = dev_fabs<T>(sA[k * NB + j]);
                    if (v > rinf) rinf = v;
                }

                T target   = dev_fmax<T>(delta, dev_fabs<T>(ajj));
                T row_term = rinf * rinf * b2inv;
                if (row_term > target) target = row_term;

                sA[j * NB + j] = dev_sqrt<T>(target);
                *info = j + 1;
                dot_scratch[0] = sA[j * NB + j];
            }
            __syncthreads();
        } else {
            if (tid == 0) {
                sA[j * NB + j] = dev_sqrt<T>(ajj);
                dot_scratch[0] = sA[j * NB + j];
            }
            __syncthreads();
        }

        T ljj = dot_scratch[0];

        for (int i = j + 1 + tid; i < NB; i += blockDim.x) {
            T sum = T(0);
            for (int k = 0; k < j; k++)
                sum += sA[k * NB + i] * sA[k * NB + j];
            sA[j * NB + i] = (sA[j * NB + i] - sum) / ljj;
        }
        __syncthreads();
    }

    // ── 3. Write back lower triangle ─────────────────────────────────────
    for (int idx = tid; idx < n_actual * n_actual; idx += blockDim.x) {
        int col = idx / n_actual;
        int row = idx % n_actual;
        if (row >= col) {
            A[(size_t)col * lda + row] = sA[col * NB + row];
        }
    }
}
template <typename T>
void potf2_panel(T* d_A, int n, int lda, T beta, T delta,
                 int* d_info, cudaStream_t stream = 0) {
    if (n <= 0) return;
    constexpr int NB = SPOTRF_PANEL_IB;  // 32
    int nwarps = 1;
    size_t smem = (NB * NB + nwarps + NB) * sizeof(T);  // always 32×32
    potf2_panel_kernel<T><<<1, 32, smem, stream>>>(d_A, n, lda, d_info, beta, delta);
    CUDA_CHECK(cudaGetLastError());
}
// Keep the old name as an alias for float (used by non-adaptive code)
inline void spotf2_panel(float* d_A, int n, int lda, float beta, float delta,
                         int* d_info, cudaStream_t stream = 0) {
    potf2_panel<float>(d_A, n, lda, beta, delta, d_info, stream);
}


// ═══════════════════════════════════════════════════════════════════
//  Diagonal perturbation kernel for mixed-precision Cholesky
// ═══════════════════════════════════════════════════════════════════

__global__ void perturb_trailing_diagonal_kernel(
    float* __restrict__ C11,
    const float* __restrict__ L10,
    int M, int K, int ldc,
    float beta, float delta) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    float alpha = C11[(size_t)i * ldc + i];
    if (alpha > 0.0f) return;

    float rinf = 0.0f;
    for (int t = 0; t < K; t++) {
        float v = fabsf(L10[(size_t)t * ldc + i]);
        if (v > rinf) rinf = v;
    }

    float b2inv = 1.0f / (beta * beta);
    float target = fmaxf(delta, fabsf(alpha));
    float row_term = rinf * rinf * b2inv;
    if (row_term > target) target = row_term;

    C11[(size_t)i * ldc + i] += (target - alpha);
}

void perturb_trailing_diagonal(float* C11, const float* L10,
                                int M, int K, int ldc,
                                float beta,
                                float delta,
                                cudaStream_t stream = 0) {
    if (M <= 0) return;
    int bl = 256, gr = (M + bl - 1) / bl;
    perturb_trailing_diagonal_kernel<<<gr, bl, 0, stream>>>(
        C11, L10, M, K, ldc, beta, delta);
    CUDA_CHECK(cudaGetLastError());
}

// Double-precision version of perturb_trailing_diagonal
__global__ void perturb_trailing_diagonal_f64_kernel(
    double* __restrict__ C11,
    const double* __restrict__ L10,
    int M, int K, int ldc,
    double beta, double delta) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    double alpha = C11[(size_t)i * ldc + i];
    if (alpha > 0.0) return;

    double rinf = 0.0;
    for (int t = 0; t < K; t++) {
        double v = fabs(L10[(size_t)t * ldc + i]);
        if (v > rinf) rinf = v;
    }

    double b2inv = 1.0 / (beta * beta);
    double target = fmax(delta, fabs(alpha));
    double row_term = rinf * rinf * b2inv;
    if (row_term > target) target = row_term;

    C11[(size_t)i * ldc + i] = (target);
}

void perturb_trailing_diagonal_f64(double* C11, const double* L10,
                                    int M, int K, int ldc,
                                    double beta, double delta,
                                    cudaStream_t stream = 0) {
    if (M <= 0) return;
    int bl = 256, gr = (M + bl - 1) / bl;
    perturb_trailing_diagonal_f64_kernel<<<gr, bl, 0, stream>>>(
        C11, L10, M, K, ldc, beta, delta);
    CUDA_CHECK(cudaGetLastError());
}


template <typename T>
void potrf_panel(T* d_A, int n, int lda,
                 T beta, T delta,
                 cublasHandle_t blas_handle,
                 int* d_info,
                 cudaStream_t stream = 0) {
    if (n <= 0) return;

    int ib = SPOTRF_PANEL_IB;

    if (n <= ib) {
        potf2_panel<T>(d_A, n, lda, beta, delta, d_info, stream);
        return;
    }

    T one = T(1), neg_one = T(-1);

    for (int jj = 0; jj < n; jj += ib) {
        int jb = std::min(ib, n - jj);

        T* A_jj = d_A + (size_t)jj * lda + jj;

        if (jj > 0) {
            T* A_j0 = d_A + (size_t)0 * lda + jj;
            if constexpr (std::is_same_v<T, float>) {
                CUBLAS_CHECK(cublasSsyrk(blas_handle,
                    CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    jb, jj, &neg_one, A_j0, lda, &one, A_jj, lda));
            } else {
                CUBLAS_CHECK(cublasDsyrk(blas_handle,
                    CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    jb, jj, &neg_one, A_j0, lda, &one, A_jj, lda));
            }
        }

        int mr = n - jj - jb;
        if (jj > 0 && mr > 0) {
            T* A_below_j0 = d_A + (size_t)0  * lda + (jj + jb);
            T* A_j0       = d_A + (size_t)0  * lda + jj;
            T* A_below_jj = d_A + (size_t)jj * lda + (jj + jb);
            if constexpr (std::is_same_v<T, float>) {
                CUBLAS_CHECK(cublasSgemm(blas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_T, mr, jb, jj,
                    &neg_one, A_below_j0, lda, A_j0, lda,
                    &one, A_below_jj, lda));
            } else {
                CUBLAS_CHECK(cublasDgemm(blas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_T, mr, jb, jj,
                    &neg_one, A_below_j0, lda, A_j0, lda,
                    &one, A_below_jj, lda));
            }
        }

        if (jj > 0) {
            T* A_j0 = d_A + (size_t)0 * lda + jj;
            if constexpr (std::is_same_v<T, float>) {
                perturb_trailing_diagonal(A_jj, A_j0, jb, jj, lda,
                                          (float)beta, (float)delta, stream);
            } else {
                perturb_trailing_diagonal_f64(A_jj, A_j0, jb, jj, lda,
                                              (double)beta, (double)delta, stream);
            }
        }

        potf2_panel<T>(A_jj, jb, lda, beta, delta, d_info, stream);

        if (mr > 0) {
            T* A_below_jj = d_A + (size_t)jj * lda + (jj + jb);
            if constexpr (std::is_same_v<T, float>) {
                CUBLAS_CHECK(cublasStrsm(blas_handle,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    mr, jb, &one, A_jj, lda, A_below_jj, lda));
            } else {
                CUBLAS_CHECK(cublasDtrsm(blas_handle,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    mr, jb, &one, A_jj, lda, A_below_jj, lda));
            }
        }
    }
}

// Keep the old name as an alias for float (used by non-adaptive code)
inline void spotrf_panel(float* d_A, int n, int lda,
                         float beta, float delta,
                         cublasHandle_t blas_handle, int* d_info,
                         cudaStream_t stream = 0) {
    potrf_panel<float>(d_A, n, lda, beta, delta, blas_handle, d_info, stream);
}

__global__ void zero_upper_f32_kernel(float* A, int n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)n * n) return;
    int col = (int)(idx / n), row = (int)(idx % n);
    if (row < col) A[idx] = 0.0f;
}

void zero_upper_f32(float* d_A, int n) {
    size_t tot = (size_t)n * n;
    zero_upper_f32_kernel<<<(unsigned)((tot + 255) / 256), 256>>>(d_A, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void zero_upper_f64_kernel(double* A, int n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)n * n) return;
    int col = (int)(idx / n), row = (int)(idx % n);
    if (row < col) A[idx] = 0.0;
}

void zero_upper_f64(double* d_A, int n) {
    size_t tot = (size_t)n * n;
    zero_upper_f64_kernel<<<(unsigned)((tot + 255) / 256), 256>>>(d_A, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void cast_f32_to_f64_kernel(double* dst, const float* src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (double)src[i];
}

void cast_f32_to_f64(double* dst, const float* src, size_t count) {
    cast_f32_to_f64_kernel<<<(unsigned)((count + 255) / 256), 256>>>(dst, src, count);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void cast_f64_to_f32_kernel(float* dst, const double* src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (float)src[i];
}

void cast_f64_to_f32(float* dst, const double* src, size_t count) {
    cast_f64_to_f32_kernel<<<(unsigned)((count + 255) / 256), 256>>>(dst, src, count);
    CUDA_CHECK(cudaGetLastError());
}

// Accumulate a float GEMM result into a double trailing matrix:
//   dst[i] += alpha_d * src[i]
// Used when the GEMM is done in low precision (output float) but the
// trailing matrix lives in double.
__global__ void accumulate_f32_into_f64_kernel(
    double* __restrict__ dst,
    const float* __restrict__ src,
    size_t count,
    double alpha) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] += alpha * (double)src[i];
}

// dst_col_major[col*ldd + row] += alpha * src_col_major[col*lds + row]
// for the lower-triangular mr x mr block
__global__ void accumulate_f32_into_f64_matrix_kernel(
    double* __restrict__ dst,
    const float* __restrict__ src,
    int mr, int ld_dst, int ld_src,
    double alpha) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)mr * mr) return;
    int col = (int)(idx / mr);
    int row = (int)(idx % mr);
    dst[(size_t)col * ld_dst + row] += alpha * (double)src[(size_t)col * ld_src + row];
}

void accumulate_f32_into_f64_matrix(double* dst, const float* src,
                                     int mr, int ld_dst, int ld_src,
                                     double alpha, cudaStream_t stream = 0) {
    size_t tot = (size_t)mr * mr;
    accumulate_f32_into_f64_matrix_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, stream>>>(
        dst, src, mr, ld_dst, ld_src, alpha);
    CUDA_CHECK(cudaGetLastError());
}

// Cast a submatrix from double (col-major, ld=ld_src) to float (col-major, ld=ld_dst)
__global__ void cast_f64_to_f32_matrix_kernel(
    float* __restrict__ dst,
    const double* __restrict__ src,
    int rows, int cols,
    int ld_dst, int ld_src) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)rows * cols) return;
    int col = (int)(idx / rows);
    int row = (int)(idx % rows);
    dst[(size_t)col * ld_dst + row] = (float)src[(size_t)col * ld_src + row];
}

void cast_f64_to_f32_matrix(float* dst, const double* src,
                             int rows, int cols, int ld_dst, int ld_src,
                             cudaStream_t stream = 0) {
    size_t tot = (size_t)rows * cols;
    cast_f64_to_f32_matrix_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, stream>>>(
        dst, src, rows, cols, ld_dst, ld_src);
    CUDA_CHECK(cudaGetLastError());
}


// ═══════════════════════════════════════════════════════════════════
//  Machine epsilon and smallest subnormal for each precision
//
//  TF32 (TensorFloat-32) is not a storage format — data stays FP32
//  but tensor cores truncate the mantissa to 10 bits during multiply.
//  Hence:  eps_TF32 = 2^-10 ≈ 9.77e-4  (same mantissa bits as FP16)
//          subnormal_TF32 = FP32's subnormal (same exponent range)
// ═══════════════════════════════════════════════════════════════════

template <typename T>
static __forceinline__ __device__ __host__ float get_epsilon() {
    if constexpr (std::is_same_v<double, T>)           return 1.11022302e-16f;  // 2^-52 (actually double, stored as float approx)
    if constexpr (std::is_same_v<float, T>)            return 1.19209290e-07f;
    if constexpr (std::is_same_v<__half, T>)           return 9.76562500e-04f;
    if constexpr (std::is_same_v<__nv_fp8_e4m3, T>)   return 0.125f;
    return 1.19209290e-07f;
}

template <typename T>
static __forceinline__ __device__ __host__ float get_smallest_subnormal() {
    if constexpr (std::is_same_v<double, T>)           return 0.0f; // underflows to 0 in float — see note below
    if constexpr (std::is_same_v<float, T>)            return 1.40129846e-45f;
    if constexpr (std::is_same_v<__half, T>)           return 5.96046448e-08f;
    if constexpr (std::is_same_v<__nv_fp8_e4m3, T>)   return 1.95312500e-03f;
    return 1.40129846e-45f;
}

// NOTE: The double subnormal min (5e-324) is far below FP32 range, so the
// threshold computation c * subnormal^2 will be 0 in float arithmetic.
// This means the FP64 threshold effectively equals just eps_fp64, which
// is the correct behaviour: FP64 has enormous exponent range so underflow
// is never the limiting factor — only rounding matters.

// TF32 epsilon: 10-bit mantissa → 2^-10
static constexpr float TF32_EPSILON        = 9.76562500e-04f;
// TF32 uses FP32 storage, so subnormal range matches FP32
static constexpr float TF32_SUBNORMAL_MIN  = 1.40129846e-45f;

// FP64 epsilon as a double for host-side use
static constexpr double FP64_EPSILON       = 1.1102230246251565e-16;  // 2^-52
static constexpr double FP64_SUBNORMAL_MIN = 4.9406564584124654e-324; // 2^-1074

static __host__ float get_epsilon_for_prec(PrecisionChoice p) {
    switch (p) {
        case PREC_FP8:  return get_epsilon<__nv_fp8_e4m3>();
        case PREC_FP16: return get_epsilon<__half>();
        case PREC_TF32: return TF32_EPSILON;
        case PREC_FP32: return get_epsilon<float>();
        case PREC_FP64: return (float)FP64_EPSILON;  // ~1.11e-16, representable in float
        default:        return get_epsilon<float>();
    }
}

static __host__ float get_smallest_subnormal_for_prec(PrecisionChoice p) {
    switch (p) {
        case PREC_FP8:  return get_smallest_subnormal<__nv_fp8_e4m3>();
        case PREC_FP16: return get_smallest_subnormal<__half>();
        case PREC_TF32: return TF32_SUBNORMAL_MIN;
        case PREC_FP32: return get_smallest_subnormal<float>();
        case PREC_FP64: return 0.0f;  // 5e-324 underflows to 0 in float — intentional
        default:        return get_smallest_subnormal<float>();
    }
}

template <typename T>
static __forceinline__ __device__ __host__ float get_beta_for_precision() {
    if constexpr (std::is_same_v<double, T>) return 1.0f;
    if constexpr (std::is_same_v<float, T>) return 1.0f;
    if constexpr (std::is_same_v<__half, T>) return 1.0f;
    if constexpr (std::is_same_v<__nv_fp8_e4m3, T>) return 9.76562500e-04f*16.0f;
    return 1.0f;
}


// ═══════════════════════════════════════════════════════════════════
//  Adaptive precision selection kernel  (boolean AND reduction)
//
//  Now tracks 5 precision levels with a 5-bit mask:
//    bit 0 (0x01) = FP8,   bit 1 (0x02) = FP16,
//    bit 2 (0x04) = TF32,  bit 3 (0x08) = FP32,
//    bit 4 (0x10) = FP64
// ═══════════════════════════════════════════════════════════════════

__global__ void update_diag_and_check_kernel(
    double* __restrict__ updated_diag,
    const double* __restrict__ orig_diag,
    const double* __restrict__ L10,
    int trailing_start,
    int M,
    int K,
    int ldc,
    float threshold_fp8,
    float threshold_fp16,
    float threshold_tf32,
    float threshold_fp32,
    float threshold_fp64,
    float eps_prime,
    unsigned int* __restrict__ block_masks) {

    extern __shared__ unsigned int smask[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int local_mask = 0x1Fu;  // all 5 bits set

    if (gid < M) {
        int diag_idx = trailing_start + gid;

        double sum_sq = 0.0;
        for (int k = 0; k < K; k++) {
            double v = L10[(size_t)k * ldc + gid];
            sum_sq += v * v;
        }
        double new_diag = updated_diag[diag_idx] - sum_sq;
        updated_diag[diag_idx] = new_diag;

        double rhs = (double)eps_prime * orig_diag[diag_idx];

        if (!((double)threshold_fp8  * new_diag < rhs)) local_mask &= ~0x01u;
        if (!((double)threshold_fp16 * new_diag < rhs)) local_mask &= ~0x02u;
        if (!((double)threshold_tf32 * new_diag < rhs)) local_mask &= ~0x04u;
        if (!((double)threshold_fp32 * new_diag < rhs)) local_mask &= ~0x08u;
        if (!((double)threshold_fp64 * new_diag < rhs)) local_mask &= ~0x10u;
    }

    smask[tid] = local_mask;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smask[tid] &= smask[tid + s];
        __syncthreads();
    }

    if (tid == 0) block_masks[blockIdx.x] = smask[0];
}

__global__ void reduce_and_kernel(
    const unsigned int* __restrict__ block_masks,
    int num_blocks,
    unsigned int* __restrict__ result) {

    extern __shared__ unsigned int smask[];
    int tid = threadIdx.x;

    unsigned int local_mask = 0x1Fu;  // all 5 bits set
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        local_mask &= block_masks[i];
    }
    smask[tid] = local_mask;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smask[tid] &= smask[tid + s];
        __syncthreads();
    }

    if (tid == 0) *result = smask[0];
}

__global__ void extract_diagonal_f64_kernel(double* __restrict__ diag,
                                            const double* __restrict__ A,
                                            int n, int lda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) diag[i] = A[(size_t)i * lda + i];
}

void extract_diagonal_f64(double* d_diag, const double* d_A, int n, int lda,
                          cudaStream_t stream = 0) {
    int bl = 256, gr = (n + bl - 1) / bl;
    extract_diagonal_f64_kernel<<<gr, bl, 0, stream>>>(d_diag, d_A, n, lda);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void extract_diagonal_kernel(float* __restrict__ diag,
                                        const float* __restrict__ A,
                                        int n, int lda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) diag[i] = A[(size_t)i * lda + i];
}

void extract_diagonal(float* d_diag, const float* d_A, int n, int lda,
                      cudaStream_t stream = 0) {
    int bl = 256, gr = (n + bl - 1) / bl;
    extract_diagonal_kernel<<<gr, bl, 0, stream>>>(d_diag, d_A, n, lda);
    CUDA_CHECK(cudaGetLastError());
}

PrecisionChoice select_precision_for_step_f64(
    double* d_updated_diag,
    const double* d_orig_diag,
    const double* d_L10,
    int trailing_start,
    int M,
    int K,
    int ldc,
    float c,
    float eps_prime,
    bool use_mx,
    unsigned int* d_block_masks,
    unsigned int* d_result_mask,
    cudaStream_t stream = 0) {

    if (M <= 0) return PREC_FP64;

    float threshold_fp8  = get_epsilon_for_prec(PREC_FP8)
                         * (1.0f + c * get_smallest_subnormal_for_prec(PREC_FP8)
                                     * get_smallest_subnormal_for_prec(PREC_FP8));
    float threshold_fp16 = get_epsilon_for_prec(PREC_FP16)
                         * (1.0f + c * get_smallest_subnormal_for_prec(PREC_FP16)
                                     * get_smallest_subnormal_for_prec(PREC_FP16));
    float threshold_tf32 = get_epsilon_for_prec(PREC_TF32)
                         * (1.0f + c * get_smallest_subnormal_for_prec(PREC_TF32)
                                     * get_smallest_subnormal_for_prec(PREC_TF32));
    float threshold_fp32 = get_epsilon_for_prec(PREC_FP32)
                         * (1.0f + c * get_smallest_subnormal_for_prec(PREC_FP32)
                                     * get_smallest_subnormal_for_prec(PREC_FP32));
    float threshold_fp64 = get_epsilon_for_prec(PREC_FP64)
                         * (1.0f + c * get_smallest_subnormal_for_prec(PREC_FP64)
                                     * get_smallest_subnormal_for_prec(PREC_FP64));
    // Note: threshold_fp64 ≈ eps_fp64 since subnormal_fp64^2 = 0 in float

    int block_size = 256;
    int num_blocks = (M + block_size - 1) / block_size;

    update_diag_and_check_kernel<<<num_blocks, block_size,
                                   block_size * sizeof(unsigned int), stream>>>(
        d_updated_diag, d_orig_diag, d_L10,
        trailing_start, M, K, ldc,
        threshold_fp8, threshold_fp16, threshold_tf32, threshold_fp32,
        threshold_fp64,
        eps_prime, d_block_masks);
    CUDA_CHECK(cudaGetLastError());

    int reduce_threads = std::min(num_blocks, 256);
    int rt_pow2 = 1;
    while (rt_pow2 < reduce_threads) rt_pow2 <<= 1;
    reduce_and_kernel<<<1, rt_pow2, rt_pow2 * sizeof(unsigned int), stream>>>(
        d_block_masks, num_blocks, d_result_mask);
    CUDA_CHECK(cudaGetLastError());

    unsigned int mask = 0;
    CUDA_CHECK(cudaMemcpyAsync(&mask, d_result_mask, sizeof(unsigned int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Pick lowest-precision level whose bit is still set
    if (mask & 0x01u) return PREC_FP8;
    if (mask & 0x02u) return PREC_FP16;
    if (mask & 0x04u) return PREC_TF32;
    if (mask & 0x08u) return PREC_FP32;
    return PREC_FP64;
}

// Keep the FP32-max version for the non-adaptive paths
PrecisionChoice select_precision_for_step(
    float* d_updated_diag,
    const float* d_orig_diag,
    const float* d_L10,
    int trailing_start,
    int M,
    int K,
    int ldc,
    float c,
    float eps_prime,
    bool use_mx,
    unsigned int* d_block_masks,
    unsigned int* d_result_mask,
    cudaStream_t stream = 0) {

    if (M <= 0) return PREC_FP32;

    float threshold_fp8  = get_epsilon_for_prec(PREC_FP8)
                         * (1.0f + c * get_smallest_subnormal_for_prec(PREC_FP8)
                                     * get_smallest_subnormal_for_prec(PREC_FP8));
    float threshold_fp16 = get_epsilon_for_prec(PREC_FP16)
                         * (1.0f + c * get_smallest_subnormal_for_prec(PREC_FP16)
                                     * get_smallest_subnormal_for_prec(PREC_FP16));
    float threshold_tf32 = get_epsilon_for_prec(PREC_TF32)
                         * (1.0f + c * get_smallest_subnormal_for_prec(PREC_TF32)
                                     * get_smallest_subnormal_for_prec(PREC_TF32));
    float threshold_fp32 = get_epsilon_for_prec(PREC_FP32)
                         * (1.0f + c * get_smallest_subnormal_for_prec(PREC_FP32)
                                     * get_smallest_subnormal_for_prec(PREC_FP32));

    int block_size = 256;
    int num_blocks = (M + block_size - 1) / block_size;

    // Reuse the 5-threshold kernel with fp64 threshold = 0 (bit always cleared)
    // Actually, we need the old 4-threshold kernel or a wrapper.
    // For simplicity, just use the same kernel with a dummy fp64 threshold:
    float threshold_fp64_dummy = 0.0f;  // will never pass → bit 4 always cleared

    // But the kernel expects double* now for the f64 version...
    // This function still works with float* — keep the old kernel signature.
    // We'll just not use it for the adaptive-f64 path.

    // For now, fall back to a simple implementation:
    // (The old kernel expected float* diag — this is only used by non-adaptive code)
    return PREC_FP32;  // placeholder — this path unused for adaptive-f64
}


// ═══════════════════════════════════════════════════════════════════
//  Ill-conditioned SPD matrix generation (FP64)
// ═══════════════════════════════════════════════════════════════════

void generate_spd(double* d_A, int n, int lda, double cond, eigenvalue_dist_t dist,
                  cublasHandle_t cb, cusolverDnHandle_t cs, cudaStream_t stream,
                  double shape_p = 2.0, int num_clusters = 4) {
    size_t nn = (size_t)n * n;
    double* d_B;
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(double) * nn));

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42ULL));
    CURAND_CHECK(curandSetStream(gen, stream));
    CURAND_CHECK(curandGenerateNormalDouble(gen, d_B, nn, 0.0, 1.0));
    CURAND_CHECK(curandDestroyGenerator(gen));

    double* d_tau;
    CUDA_CHECK(cudaMalloc(&d_tau, sizeof(double) * n));
    int gws = 0, ows = 0;
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cs, n, n, d_B, n, &gws));
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cs, n, n, n, d_B, n, d_tau, &ows));
    int ws = std::max(gws, ows);
    double* d_w;
    CUDA_CHECK(cudaMalloc(&d_w, sizeof(double) * ws));
    int* d_i;
    CUDA_CHECK(cudaMalloc(&d_i, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDgeqrf(cs, n, n, d_B, n, d_tau, d_w, gws, d_i));
    CUSOLVER_CHECK(cusolverDnDorgqr(cs, n, n, n, d_B, n, d_tau, d_w, ows, d_i));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_i));

    {
        std::vector<double> eigs(n);
        double ic = 1.0 / cond;
        switch (dist) {
            case DIST_GEOMETRIC:
                for (int k = 0; k < n; k++)
                    eigs[k] = std::pow(cond, -(double)k / (n - 1));
                break;
            case DIST_ARITHMETIC:
                for (int k = 0; k < n; k++)
                    eigs[k] = 1.0 - (double)k / (n - 1) * (1.0 - ic);
                break;
            case DIST_HALF_HALF:
                for (int k = 0; k < n; k++)
                    eigs[k] = (k < n / 2) ? 1.0 : ic;
                break;
            case DIST_RANDOM:
                srand(123);
                eigs[0] = 1.0;
                for (int k = 1; k < n - 1; k++)
                    eigs[k] = ic + ((double)rand() / RAND_MAX) * (1.0 - ic);
                eigs[n - 1] = ic;
                break;
            case DIST_POLY_DECAY:
                for (int k = 0; k < n; k++) {
                    double t = (double)k / (n - 1);
                    eigs[k] = (1.0 - std::pow(t, shape_p)) * (1.0 - ic) + ic;
                }
                break;
            case DIST_MULTI_CLUSTER: {
                int nc = std::max(2, num_clusters);
                int per_cluster = n / nc;
                int remainder = n - per_cluster * nc;
                int idx = 0;
                for (int c = 0; c < nc; c++) {
                    double val = std::pow(cond, -(double)c / (nc - 1));
                    int count = per_cluster + (c < remainder ? 1 : 0);
                    for (int j = 0; j < count && idx < n; j++)
                        eigs[idx++] = val;
                }
                break;
            }
            case DIST_POWER_LAW:
                for (int k = 0; k < n; k++) {
                    double t = (double)k / (n - 1);
                    eigs[k] = std::pow(cond, -std::pow(t, shape_p));
                }
                break;
            default:
                fprintf(stderr, "Unknown distribution\n");
                exit(1);
        }
        int bl = 256, gr = (n + bl - 1) / bl;
        for (int k = 0; k < n; k++) {
            double s = std::sqrt(eigs[k]);
            if (std::abs(s - 1.0) > 1e-15)
                scale_col_kernel<<<gr, bl, 0, stream>>>(d_B, n, n, k, s);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    double one = 1.0, zero = 0.0;
    CUBLAS_CHECK(cublasDgemm(cb, CUBLAS_OP_N, CUBLAS_OP_T,
                             n, n, n, &one, d_B, n, d_B, n, &zero, d_A, lda));
    CUDA_CHECK(cudaFree(d_B));
}


// ═══════════════════════════════════════════════════════════════════
//  Conversion kernels — FP32 column-major → low-precision column-major
//
//  All storage is column-major throughout.  The panel L10 lives in the
//  lower-left of the full column-major matrix (ld = lda = n).  We
//  convert it into a packed column-major buffer with ld_dst = rows.
// ═══════════════════════════════════════════════════════════════════


#define FP8_E4M3_MAX 448.0f

__global__ void fp32_to_fp8_tiled_kernel(
    __nv_fp8_e4m3* __restrict__ dst,
    const float*   __restrict__ src,
    float*         __restrict__ scales,   // [grid.x][grid.y] output
    int rows, int cols,
    int ld_src, int ld_dst)
{
    // ── shared memory ────────────────────────────────────────────────────
    __shared__ float tile[QUANT_TILE][QUANT_TILE + 1];  // +1: avoid bank conflicts
    __shared__ float smem_max[QUANT_TILE];               // one slot per row of threads

    int row_base = blockIdx.x * QUANT_TILE;
    int col_base = blockIdx.y * QUANT_TILE;
    int tr = threadIdx.x;   // row within tile
    int tc = threadIdx.y;   // col within tile

    // ── 1. Coalesced load (col-major src → transposed shared mem) ────────
    int src_row = row_base + tr;
    int src_col = col_base + tc;

    float v = 0.0f;
    if (src_row < rows && src_col < cols)
        v = src[(size_t)src_col * ld_src + src_row];

    tile[tc][tr] = v;   // transpose: tile[col][row] so later reads are coalesced
    __syncthreads();

    float local_abs = fabsf(tile[tr][tc]);

    #pragma unroll
    for (int offset = QUANT_TILE / 2; offset > 0; offset >>= 1)
        local_abs = fmaxf(local_abs, __shfl_xor_sync(0xffffffff, local_abs, offset));

    // Thread tc==0 writes the per-row max
    if (tc == 0)
        smem_max[tr] = local_abs;
    __syncthreads();

    // Stage B: a single warp (tr < 32, tc == 0) reduces smem_max[]
    float tile_max = 0.0f;
    if (tc == 0) {
        tile_max = smem_max[tr];
        #pragma unroll
        for (int offset = QUANT_TILE / 2; offset > 0; offset >>= 1)
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, offset));

        // tr==0, tc==0 writes the scale
        if (tr == 0) {
            float scale = (tile_max > 9.76562500e-04f*16.0f) ? tile_max / FP8_E4M3_MAX : 1.0f;
            int scale_cols = (cols + QUANT_TILE - 1) / QUANT_TILE;
            scales[blockIdx.x * scale_cols + blockIdx.y] = scale;
            smem_max[0] = scale;   // broadcast scale for the write stage
        }
    }
    __syncthreads();

    float scale = smem_max[0];          // every thread reads the final scale
    float inv_scale = 1.0f / scale;

    int dst_row = row_base + tr;
    int dst_col = col_base + tc;

    if (dst_row < rows && dst_col < cols) {
        float val = tile[tr][tc] * inv_scale;
        val = fminf(fmaxf(val, -FP8_E4M3_MAX/16.0f), FP8_E4M3_MAX/16.0f);
        dst[(size_t)dst_row * ld_dst + dst_col] = __nv_fp8_e4m3(val);
    }

    if (row_base == 0 && col_base == 0 && tr == 0 && tc == 0)
        scales[0] = -scales[0];
}
 __global__ void fp32_to_fp16_tiled_kernel(
    __half* __restrict__      dst,
    const float* __restrict__ src,
    int rows, int cols,
    int ld_src, int ld_dst)
{
    __shared__ float tile[QUANT_TILE][QUANT_TILE + 1];

    int row_base = blockIdx.x * QUANT_TILE;
    int col_base = blockIdx.y * QUANT_TILE;
    int tr = threadIdx.x, tc = threadIdx.y;

    int src_row = row_base + tr;
    int src_col = col_base + tc;

    float v = 0.0f;
    if (src_row < rows && src_col < cols)
        v = src[(size_t)src_col * ld_src + src_row];
    tile[tc][tr] = v;
    __syncthreads();

    int dst_row = row_base + tc;
    int dst_col = col_base + tr;

    if (dst_row < rows && dst_col < cols)
        dst[(size_t)dst_row * ld_dst + dst_col] = (__float2half(fminf(fmaxf(v, -65504.f), 65504.f)));
}
 
// Drop-in replacements for launch_convert
template <typename T>
void launch_convert(T* dst, const float* src, float* d_scales, 
                          int rows, int cols, int ld_src, int ld_dst,
                          cudaStream_t stream = 0) {
    dim3 grid((rows + QUANT_TILE - 1) / QUANT_TILE,
              (cols + QUANT_TILE - 1) / QUANT_TILE);
    dim3 block(QUANT_TILE, QUANT_TILE);

    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
        fp32_to_fp8_tiled_kernel<<<grid, block, 0, stream>>>(
            dst, src, d_scales, rows, cols, ld_src, ld_dst);
    else
        fp32_to_fp16_tiled_kernel<<<grid, block, 0, stream>>>(
            dst, src, rows, cols, ld_src, ld_dst);
        CUDA_CHECK(cudaGetLastError());
}
 


// ═══════════════════════════════════════════════════════════════════
//  CUTLASS GEMM type definitions
// ═══════════════════════════════════════════════════════════════════

using CholClusterShape = Shape<_1, _1, _1>;

// --- FP8 (tensorwise, no per-block scaling) ---
using Fp8TileShape = Shape<_128, _128, _128>;

using Fp8CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Fp8TileShape, CholClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    float, cutlass::layout::ColumnMajor, 4,
    float, cutlass::layout::ColumnMajor, 4,
    cutlass::epilogue::TmaWarpSpecializedCooperative
>::CollectiveOp;

using Fp8CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
    float, Fp8TileShape, CholClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Fp8CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum
>::CollectiveOp;

using Fp8GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, Fp8CollectiveMainloop, Fp8CollectiveEpilogue>;
using Fp8CutlassGemm = cutlass::gemm::device::GemmUniversalAdapter<Fp8GemmKernel>;

// --- FP16 ---
using Fp16TileShape = Shape<_128, _128, _64>;

using Fp16CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Fp16TileShape, CholClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    float, cutlass::layout::ColumnMajor, 4,
    float, cutlass::layout::ColumnMajor, 4,
    cutlass::epilogue::TmaWarpSpecializedCooperative
>::CollectiveOp;

using Fp16CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    float, Fp16TileShape, CholClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Fp16CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using Fp16GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, Fp16CollectiveMainloop, Fp16CollectiveEpilogue>;
using Fp16CutlassGemm = cutlass::gemm::device::GemmUniversalAdapter<Fp16GemmKernel>;

// --- Groupwise-scaled FP8 ---
using GsFp8TileShape = Shape<_128, _128, _128>;

constexpr int GsScaleGranularityM = 128;
constexpr int GsScaleGranularityN = 128;
constexpr int GsScaleGranularityK = 128;

using GsScaleConfig  = decltype(cutlass::detail::sm90_trivial_blockwise_scale_config(GsFp8TileShape{}));
using GsLayoutSFA    = decltype(GsScaleConfig::deduce_layoutSFA());
using GsLayoutSFB    = decltype(GsScaleConfig::deduce_layoutSFB());

using GsKernelSchedule  = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
using GsEpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

using GsFp8CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    GsFp8TileShape, CholClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    float, cutlass::layout::ColumnMajor, 4,
    float, cutlass::layout::ColumnMajor, 4,
    GsEpilogueSchedule
>::CollectiveOp;

using GsFp8CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cute::tuple<cutlass::layout::RowMajor, GsLayoutSFA>, 16,
    cutlass::float_e4m3_t, cute::tuple<cutlass::layout::ColumnMajor, GsLayoutSFB>, 16,
    float, GsFp8TileShape, CholClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename GsFp8CollectiveEpilogue::SharedStorage))>,
    GsKernelSchedule
>::CollectiveOp;

using GsFp8GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, GsFp8CollectiveMainloop, GsFp8CollectiveEpilogue>;
using GsFp8CutlassGemm = cutlass::gemm::device::GemmUniversalAdapter<GsFp8GemmKernel>;


// ═══════════════════════════════════════════════════════════════════
//  CUTLASS GEMM runners — generic (FP16, FP8 tensorwise)
//
//  Takes external workspace to avoid cudaMalloc/Free per call.
// ═══════════════════════════════════════════════════════════════════
template <typename CutlassGemmT, bool AlphaOnDevice = false>
void run_cutlass_gemm(int M, int N, int K,
                      const void* d_A, const void* d_B,
                      float* d_C, int ldc,
                      std::conditional_t<AlphaOnDevice, const float*, float> alpha,
                      float beta,
                      void* workspace,
                      cudaStream_t stream = 0)
{
    using StrideA  = typename CutlassGemmT::GemmKernel::StrideA;
    using StrideB  = typename CutlassGemmT::GemmKernel::StrideB;
    using StrideC  = typename CutlassGemmT::GemmKernel::StrideC;
    using StrideD  = typename CutlassGemmT::GemmKernel::StrideD;
    using ElementA = typename CutlassGemmT::GemmKernel::ElementA;
    using ElementB = typename CutlassGemmT::GemmKernel::ElementB;

    StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC sc = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    if (ldc != M) {
        cute::get<1>(sc) = ldc;
        cute::get<1>(sd) = ldc;
    }

    // Construct args with zeroed epilogue thread fields, fill them below.
    // Avoids EpilogueOutputOp::Params which is a CUTLASS 2.x concept —
    // in CUTLASS 3.x the fusion op (LinearCombination) exposes ::Arguments,
    // accessible via args.epilogue.thread.
    typename CutlassGemmT::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
        { reinterpret_cast<const ElementA*>(d_A), sa,
          reinterpret_cast<const ElementB*>(d_B), sb },
        { {}, d_C, sc, d_C, sd }
    };

    auto& th = args.epilogue.thread;   // cutlass::epilogue::fusion::LinearCombination::Arguments
    th.beta     = beta;
    th.beta_ptr = nullptr;             // beta always from host

    if constexpr (AlphaOnDevice) {
        th.alpha     = 1.0f;           // ignored when alpha_ptr != nullptr
        th.alpha_ptr = alpha;          // alpha is const float* here
    } else {
        th.alpha     = alpha;          // alpha is float here
        th.alpha_ptr = nullptr;
    }

    CutlassGemmT op;
#ifndef NDEBUG
    CUTLASS_CHECK(op.can_implement(args));
#endif
    CUTLASS_CHECK(op.initialize(args, workspace, stream));
    CUTLASS_CHECK(op.run(stream));
}

// ═══════════════════════════════════════════════════════════════════
//  CUTLASS GEMM runner — blockwise-scaled FP8
// ═══════════════════════════════════════════════════════════════════

inline void run_cutlass_gemm_gsfp8(
        int M, int N, int K,
        const __nv_fp8_e4m3* d_A,
        const __nv_fp8_e4m3* d_B,
        const float*          d_sfa,
        const float*          d_sfb,
        float*                d_C,
        int                   ldc,
        float                 alpha,
        float                 beta,
        void*                 workspace,
        cudaStream_t          stream = 0)
{
    using GemmOp  = GsFp8CutlassGemm;
    using Fp8     = cutlass::float_e4m3_t;
    using StrideA = typename GemmOp::GemmKernel::StrideA;
    using StrideB = typename GemmOp::GemmKernel::StrideB;
    using StrideC = typename GemmOp::GemmKernel::StrideC;
    using StrideD = typename GemmOp::GemmKernel::StrideD;

    StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    StrideC sc = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    if (ldc != M) {
        cute::get<1>(sc) = ldc;
        cute::get<1>(sd) = ldc;
    }

    auto lsfa = GsScaleConfig::tile_atom_to_shape_SFA(
                    cute::make_shape(M, N, K, 1));
    auto lsfb = GsScaleConfig::tile_atom_to_shape_SFB(
                    cute::make_shape(M, N, K, 1));

    typename GemmOp::Arguments args {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { reinterpret_cast<const Fp8*>(d_A), sa,
          reinterpret_cast<const Fp8*>(d_B), sb,
          d_sfa, lsfa,
          d_sfb, lsfb },
        { {alpha, beta}, d_C, sc, d_C, sd }
    };

    GemmOp op;
#ifndef NDEBUG
    CUTLASS_CHECK(op.can_implement(args));
#endif
    CUTLASS_CHECK(op.initialize(args, workspace, stream));
    CUTLASS_CHECK(op.run(stream));
}

// ═══════════════════════════════════════════════════════════════════
//  Non-template wrappers for CUTLASS GEMMs
//  (Avoids dependent-name parsing errors when called from templates)
// ═══════════════════════════════════════════════════════════════════

inline void run_fp16_gemm(int M, int N, int K, const void* A, const void* B,
                           float* C, int ldc, float alpha, float beta,
                           void* ws, cudaStream_t stream = 0) {
    run_cutlass_gemm<Fp16CutlassGemm>(M, N, K, A, B, C, ldc, alpha, beta, ws, stream);
}

inline void run_fp8_gemm_scaled(int M, int N, int K, const void* A, const void* B,
                                 float* C, int ldc, const float* alpha, float beta,
                                 void* ws, cudaStream_t stream = 0) {
    run_cutlass_gemm<Fp8CutlassGemm, true>(M, N, K, A, B, C, ldc, alpha, beta, ws, stream);
}

inline void run_fp8_gemm(int M, int N, int K, const void* A, const void* B,
                           float* C, int ldc, float alpha, float beta,
                           void* ws, cudaStream_t stream = 0) {
    run_cutlass_gemm<Fp8CutlassGemm>(M, N, K, A, B, C, ldc, alpha, beta, ws, stream);
}


// ═══════════════════════════════════════════════════════════════════
//  Persistent GEMM launchers — for repeated same-shape calls
// ═══════════════════════════════════════════════════════════════════

// Generic launcher for FP16 / FP8 tensorwise (no scale factors)
template <typename CutlassGemmT>
class CutlassGemmLauncher {
public:
    using GemmOp   = CutlassGemmT;
    using ElementA = typename GemmOp::GemmKernel::ElementA;
    using ElementB = typename GemmOp::GemmKernel::ElementB;
    using StrideA  = typename GemmOp::GemmKernel::StrideA;
    using StrideB  = typename GemmOp::GemmKernel::StrideB;
    using StrideC  = typename GemmOp::GemmKernel::StrideC;
    using StrideD  = typename GemmOp::GemmKernel::StrideD;

    void configure(int M, int N, int K, int ldc,
                   float alpha, float beta,
                   void* workspace, cudaStream_t stream = 0) {
        M_ = M; N_ = N; K_ = K; ldc_ = ldc;
        alpha_ = alpha; beta_ = beta; workspace_ = workspace;
        sa_ = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        sb_ = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        sc_ = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        sd_ = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
        if (ldc != M) { cute::get<1>(sc_) = ldc; cute::get<1>(sd_) = ldc; }
#ifndef NDEBUG
        auto args = make_args(nullptr, nullptr, nullptr);
        CUTLASS_CHECK(op_.can_implement(args));
#endif
        configured_ = true;
    }

    void run(const void* d_A, const void* d_B, float* d_C,
             cudaStream_t stream = 0) {
        assert(configured_ && "call configure() first");
        auto args = make_args(d_A, d_B, d_C);
        CUTLASS_CHECK(op_.initialize(args, workspace_, stream));
        CUTLASS_CHECK(op_.run(stream));
    }

private:
    typename GemmOp::Arguments make_args(
            const void* d_A, const void* d_B, float* d_C) const {
        return {
            cutlass::gemm::GemmUniversalMode::kGemm, {M_, N_, K_, 1},
            { reinterpret_cast<const ElementA*>(d_A), sa_,
              reinterpret_cast<const ElementB*>(d_B), sb_ },
            { {alpha_, beta_}, d_C, sc_, d_C, sd_ }
        };
    }
    GemmOp op_{}; int M_{}, N_{}, K_{}, ldc_{};
    float alpha_{}, beta_{}; void* workspace_{};
    StrideA sa_{}; StrideB sb_{}; StrideC sc_{}; StrideD sd_{};
    bool configured_{false};
};

// Blockwise-scaled FP8 launcher
class GsFp8GemmLauncher {
public:
    using GemmOp  = GsFp8CutlassGemm;
    using Fp8     = cutlass::float_e4m3_t;
    using StrideA = typename GemmOp::GemmKernel::StrideA;
    using StrideB = typename GemmOp::GemmKernel::StrideB;
    using StrideC = typename GemmOp::GemmKernel::StrideC;
    using StrideD = typename GemmOp::GemmKernel::StrideD;

    void configure(int M, int N, int K, int ldc,
                   float alpha, float beta,
                   void* workspace, cudaStream_t stream = 0) {
        M_ = M; N_ = N; K_ = K; ldc_ = ldc;
        alpha_ = alpha; beta_ = beta; workspace_ = workspace;
        sa_ = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        sb_ = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        sc_ = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        sd_ = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
        if (ldc != M) { cute::get<1>(sc_) = ldc; cute::get<1>(sd_) = ldc; }
        lsfa_ = GsScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
        lsfb_ = GsScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
#ifndef NDEBUG
        auto args = make_args(nullptr, nullptr, nullptr, nullptr, nullptr);
        CUTLASS_CHECK(op_.can_implement(args));
#endif
        configured_ = true;
    }

    void run(const __nv_fp8_e4m3* d_A, const __nv_fp8_e4m3* d_B,
             const float* d_sfa, const float* d_sfb,
             float* d_C, cudaStream_t stream = 0) {
        assert(configured_ && "call configure() first");
        auto args = make_args(d_A, d_B, d_sfa, d_sfb, d_C);
        CUTLASS_CHECK(op_.initialize(args, workspace_, stream));
        CUTLASS_CHECK(op_.run(stream));
    }

private:
    typename GemmOp::Arguments make_args(
            const __nv_fp8_e4m3* d_A, const __nv_fp8_e4m3* d_B,
            const float* d_sfa, const float* d_sfb, float* d_C) const {
        return {
            cutlass::gemm::GemmUniversalMode::kGemm, {M_, N_, K_, 1},
            { reinterpret_cast<const Fp8*>(d_A), sa_,
              reinterpret_cast<const Fp8*>(d_B), sb_,
              d_sfa, lsfa_, d_sfb, lsfb_ },
            { {alpha_, beta_}, d_C, sc_, d_C, sd_ }
        };
    }
    GemmOp op_{}; int M_{}, N_{}, K_{}, ldc_{};
    float alpha_{}, beta_{}; void* workspace_{};
    StrideA sa_{}; StrideB sb_{}; StrideC sc_{}; StrideD sd_{};
    decltype(GsScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(0,0,0,0))) lsfa_{};
    decltype(GsScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(0,0,0,0))) lsfb_{};
    bool configured_{false};
};


// ── Warp-level max using shuffle (no shared memory needed) ──────────────────
__device__ __forceinline__ float warp_reduce_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  1));
    return v;
}

// ── Warp-per-tile: each warp owns one (bm,bk) tile ──────────────────────────
// Launch: grid( ceil(n_tiles / WARPS_PER_CTA) ), threads( WARPS_PER_CTA * 32 )
// One CTA covers WARPS_PER_CTA tiles simultaneously.
template<int WARPS_PER_CTA>
__global__ void fp32_to_fp8_blockscaled_warp_per_tile(
    __nv_fp8_e4m3* __restrict__ dst,
    float*         __restrict__ scales,
    const float*   __restrict__ src,
    int rows, int cols,
    int ld_src, int ld_dst,
    int scale_block_m, int scale_block_k,
    int n_scale_m,     int n_scale_k)
{
    const int lane    = threadIdx.x & 31;
    const int wid     = threadIdx.x >> 5;
    const int tile_id = blockIdx.x * WARPS_PER_CTA + wid;
    const int n_tiles = n_scale_m * n_scale_k;

    if (tile_id >= n_tiles) return;

    const int bm = tile_id % n_scale_m;
    const int bk = tile_id / n_scale_m;

    const int row_start  = bm * scale_block_m;
    const int col_start  = bk * scale_block_k;
    const int row_end    = min(row_start + scale_block_m, rows);
    const int col_end    = min(col_start + scale_block_k, cols);
    const int block_rows = row_end - row_start;
    const int block_cols = col_end - col_start;
    const int block_elems = block_rows * block_cols;

    // ── Pass 1: amax via pure warp shuffle — zero smem ─────────────────────
    float local_max = 0.f;
    for (int i = lane; i < block_elems; i += 32) {
        const int row_local = i % block_rows;
        const int col_local = i / block_rows;
        local_max = fmaxf(local_max,
            fabsf(src[(size_t)(col_start + col_local) * ld_src
                             + (row_start + row_local)]));
    }
    // All 32 lanes get the warp max (every shfl_xor step is collective)
    local_max = warp_reduce_max(local_max);

    const float scale     = (local_max > 9.76562500e-04f*16.0f) ? (local_max / 448.f) : 1.f;
    const float inv_scale = 1.f / scale;

    if (lane == 0) scales[bk * n_scale_m + bm] = scale;

    // ── Pass 2: quantize + coalesced write ──────────────────────────────────
    for (int i = lane; i < block_elems; i += 32) {
        const int row_local = i / block_cols;
        const int col_local = i % block_cols;
        float v = src[(size_t)(col_start + col_local) * ld_src
                             + (row_start + row_local)] * inv_scale;
        v = fminf(fmaxf(v, -448.f), 448.f);
        dst[(size_t)(row_start + row_local) * ld_dst
                   + (col_start + col_local)] = __nv_fp8_e4m3(v);
    }
}

__global__ void hasNaN_kernel(const double* __restrict__ data, int n, int lda, int* has_nan) {
    // Each thread handles one (row, col) and strides through rows
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int stride_row = blockDim.y * gridDim.y;
    int stride_col = blockDim.x * gridDim.x;

    bool found = false;
    for (int r = row; r < n && !found; r += stride_row) {
        for (int c = col; c < n && !found; c += stride_col) {
            if (isnan(data[r * lda + c])) found = true;
        }
        if (__any_sync(0xFFFFFFFF, *has_nan)) return;
    }

    if (__any_sync(0xFFFFFFFF, found)) {
        if (threadIdx.x % 32 == 0) atomicExch(has_nan, 1);
    }
}

// ── Variant for small grids (total tiles < threshold) ───────────────────────
// No data cache in smem → only NUM_WARPS*4 bytes reserved → max occupancy.
// Two global passes are fine: the total data is tiny.
template<int BLOCK_DIM>
__global__ void fp32_to_fp8_blockscaled_kernel_small(
    __nv_fp8_e4m3* __restrict__ dst,
    float*         __restrict__ scales,
    const float*   __restrict__ src,
    int rows, int cols,
    int ld_src, int ld_dst,
    int scale_block_m, int scale_block_k,
    int n_scale_m,     int n_scale_k)
{
    constexpr int NUM_WARPS = BLOCK_DIM / 32;

    const int bm  = blockIdx.x,  bk  = blockIdx.y;
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;

    const int row_start  = bm * scale_block_m;
    const int col_start  = bk * scale_block_k;
    const int row_end    = min(row_start + scale_block_m, rows);
    const int col_end    = min(col_start + scale_block_k, cols);
    const int block_rows = row_end - row_start;
    const int block_cols = col_end - col_start;
    const int block_elems = block_rows * block_cols;

    // Only cross-warp scratch — 32 bytes for 8 warps, negligible occupancy cost
    __shared__ float warp_maxes[NUM_WARPS];

    // ── Pass 1: find amax ───────────────────────────────────────────────────
    float local_max = 0.f;
    for (int i = tid; i < block_elems; i += BLOCK_DIM) {
        const int row_local = i % block_rows;
        const int col_local = i / block_rows;
        local_max = fmaxf(local_max,
            fabsf(src[(size_t)(col_start + col_local) * ld_src
                             + (row_start + row_local)]));
    }
    local_max = warp_reduce_max(local_max);      // shuffle — no smem needed
    if (lane == 0) warp_maxes[wid] = local_max;
    __syncthreads();

    if (wid == 0) {
        float v = (lane < NUM_WARPS) ? warp_maxes[lane] : 0.f;
        v = warp_reduce_max(v);
        if (lane == 0) warp_maxes[0] = v;
    }
    __syncthreads();

    const float amax      = warp_maxes[0];
    const float scale     = (amax > 9.76562500e-04f*16.0f) ? (amax / 448.f) : 1.f;
    const float inv_scale = 1.f / scale;

    if (tid == 0) scales[bk * n_scale_m + bm] = scale;

    // ── Pass 2: quantize → dst, row-major for coalesced writes ─────────────
    // src read is strided here, but for small tile sizes L1 absorbs it cleanly
    for (int i = tid; i < block_elems; i += BLOCK_DIM) {
        const int row_local = i / block_cols;   // row-major: coalesced dst write
        const int col_local = i % block_cols;
        float v = src[(size_t)(col_start + col_local) * ld_src
                             + (row_start + row_local)] * inv_scale;
        v = fminf(fmaxf(v, -448.f), 448.f);
        dst[(size_t)(row_start + row_local) * ld_dst
                   + (col_start + col_local)] = __nv_fp8_e4m3(v);
    }
}

template<int BLOCK_DIM>
__global__ void fp32_to_fp8_blockscaled_kernel_opt(
    __nv_fp8_e4m3* __restrict__ dst,
    float*         __restrict__ scales,
    const float*   __restrict__ src,
    int rows, int cols,
    int ld_src, int ld_dst,
    int scale_block_m, int scale_block_k,
    int n_scale_m,     int n_scale_k)
{
    constexpr int NUM_WARPS = BLOCK_DIM / 32;

    const int bm  = blockIdx.x,  bk  = blockIdx.y;
    const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;

    const int row_start  = bm * scale_block_m;
    const int col_start  = bk * scale_block_k;
    const int row_end    = min(row_start + scale_block_m, rows);
    const int col_end    = min(col_start + scale_block_k, cols);
    const int block_rows = row_end - row_start;
    const int block_cols = col_end - col_start;
    const int block_elems = block_rows * block_cols;

    // +1 padding on the row dimension eliminates shared-memory bank conflicts
    // when block_rows is a multiple of 32 (e.g. 32, 64, 128).
    const int padded_rows = block_rows + 1;

    extern __shared__ float shmem[];
    float* data_cache = shmem;                        // [col * padded_rows + row]
    float* warp_maxes = shmem + padded_rows * block_cols; // [NUM_WARPS]

    // ── Pass 1: load src → shared memory, accumulate local amax ────────────
    // src is column-major: src[col * ld_src + row].
    // Iterating i = col_local * block_rows + row_local gives coalesced reads:
    // consecutive threads differ by 1 in row_local within the same column.
    float local_max = 0.0f;

    for (int i = tid; i < block_elems; i += BLOCK_DIM) {
        const int row_local = i % block_rows;
        const int col_local = i / block_rows;
        const float v = src[(size_t)(col_start + col_local) * ld_src
                            +          (row_start + row_local)];
        // Store col-major with padding to avoid bank conflicts on read-back
        data_cache[col_local * padded_rows + row_local] = v;
        local_max = fmaxf(local_max, fabsf(v));
    }

    // ── Reduction: warp shuffle → inter-warp via smem ──────────────────────
    local_max = warp_reduce_max(local_max);
    if (lane == 0) warp_maxes[wid] = local_max;
    __syncthreads();

    if (wid == 0) {
        float v = (lane < NUM_WARPS) ? warp_maxes[lane] : 0.0f;
        v = warp_reduce_max(v);
        if (lane == 0) warp_maxes[0] = v;   // broadcast result via smem
    }
    __syncthreads();

    const float amax      = warp_maxes[0];
    const float scale     = (amax > 9.76562500e-04f*16.0f) ? (amax / 448.f) : 1.f;
    const float inv_scale = 1.f / scale;

    if (tid == 0) scales[bk * n_scale_m + bm] = scale;

    // ── Pass 2: quantize from smem → dst (row-major, coalesced writes) ──────
    // Iterate row-major: row_local = i / block_cols, col_local = i % block_cols.
    // Consecutive threads share the same row_local and differ by 1 in col_local,
    // so writes to dst[row * ld_dst + col] are stride-1 → fully coalesced.
    for (int i = tid; i < block_elems; i += BLOCK_DIM) {
        const int row_local = i / block_cols;
        const int col_local = i % block_cols;

        // Smem read: col-major with padding (bank-conflict-free)
        float v = data_cache[col_local * padded_rows + row_local] * inv_scale;
        v = fminf(fmaxf(v, -448.f), 448.f);

        // Coalesced write: consecutive threads → consecutive column addresses
        dst[(size_t)(row_start + row_local) * ld_dst
                   + (col_start + col_local)] = __nv_fp8_e4m3(v);
    }
}

inline void launch_fp32_to_fp8_blockscaled(
    __nv_fp8_e4m3* dst, float* scales, const float* src,
    int rows, int cols, int ld_src, int ld_dst,
    int scale_block_m, int scale_block_k,
    int n_scale_m, int n_scale_k,
    cudaStream_t stream)
{
    constexpr int BLOCK_DIM      = 256;
    constexpr int NUM_WARPS      = BLOCK_DIM / 32;  // 8
    constexpr int WARPS_PER_CTA  = 8;               // for warp-per-tile kernel
    // Tune these thresholds based on your GPU:
    constexpr int WARP_TILE_THRESHOLD  = 8;   // ≤ 8 tiles  → warp-per-tile
    constexpr int SMALL_TILE_THRESHOLD = 32;  // ≤ 32 tiles → small kernel

    const int total_tiles = n_scale_m * n_scale_k;
    dim3 grid(n_scale_m, n_scale_k);

    if (total_tiles <= WARP_TILE_THRESHOLD) {
        // One CTA, warps split the tiles — zero smem, zero cudaFuncSetAttribute
        int ctas = (total_tiles + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
        fp32_to_fp8_blockscaled_warp_per_tile<WARPS_PER_CTA>
            <<<ctas, WARPS_PER_CTA * 32, 0, stream>>>(
                dst, scales, src, rows, cols, ld_src, ld_dst,
                scale_block_m, scale_block_k, n_scale_m, n_scale_k);

    } else if (total_tiles <= SMALL_TILE_THRESHOLD) {
        // Normal grid but no data cache — 32 bytes smem, full occupancy
        fp32_to_fp8_blockscaled_kernel_small<BLOCK_DIM>
            <<<grid, BLOCK_DIM, NUM_WARPS * sizeof(float), stream>>>(
                dst, scales, src, rows, cols, ld_src, ld_dst,
                scale_block_m, scale_block_k, n_scale_m, n_scale_k);

    }  else {
        // Large grid — full smem cache, coalesced everything
        const int padded_rows = scale_block_m + 1;
        size_t smem = (size_t(padded_rows) * scale_block_k + NUM_WARPS) * sizeof(float);

        // ← this was missing: must opt in before the 48 KB default limit is exceeded
        CUDA_CHECK(cudaFuncSetAttribute(
            fp32_to_fp8_blockscaled_kernel_opt<BLOCK_DIM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem));

        fp32_to_fp8_blockscaled_kernel_opt<BLOCK_DIM>
            <<<grid, BLOCK_DIM, smem, stream>>>(
                dst, scales, src, rows, cols, ld_src, ld_dst,
                scale_block_m, scale_block_k, n_scale_m, n_scale_k);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ═══════════════════════════════════════════════════════════════════
//  GemmWorkspace — pre-allocates all buffers for the Cholesky loop
// ═══════════════════════════════════════════════════════════════════

struct GemmWorkspace {
    void*  convert_buf;
    size_t convert_buf_bytes;

    float* scale_A;
    float* scale_B;
    size_t scale_bytes;

    void*  cutlass_ws;
    size_t cutlass_ws_bytes;

    // Temporary float buffer for GEMM output when trailing matrix is double
    float* gemm_temp;
    size_t gemm_temp_bytes;

    // Temporary float buffer for panel cast (double → float) before low-prec convert
    float* panel_f32;
    size_t panel_f32_bytes;

    void allocate(int n, int nb, bool need_f64_temps = false) {
        convert_buf_bytes = sizeof(__half) * (size_t)n * nb;
        CUDA_CHECK(cudaMalloc(&convert_buf, convert_buf_bytes));

        int max_nsm = (n + 127) / 128;
        int max_nsk = (nb + 127) / 128;
        scale_bytes = sizeof(float) * (size_t)max_nsm * max_nsk;
        CUDA_CHECK(cudaMalloc(&scale_A, scale_bytes));
        CUDA_CHECK(cudaMalloc(&scale_B, scale_bytes));

        // Compute max workspace across all three CUTLASS kernel types
        cutlass_ws_bytes = 0;

        // Block-scaled FP8
        {
            using GemmOp = GsFp8CutlassGemm;
            using StrideA = typename GemmOp::GemmKernel::StrideA;
            using StrideB = typename GemmOp::GemmKernel::StrideB;
            using StrideC = typename GemmOp::GemmKernel::StrideC;
            using StrideD = typename GemmOp::GemmKernel::StrideD;
            StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {n, nb, 1});
            StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {n, nb, 1});
            StrideC sc = cutlass::make_cute_packed_stride(StrideC{}, {n, n, 1});
            StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {n, n, 1});
            auto lsfa = GsScaleConfig::tile_atom_to_shape_SFA(make_shape(n, n, nb, 1));
            auto lsfb = GsScaleConfig::tile_atom_to_shape_SFB(make_shape(n, n, nb, 1));
            typename GemmOp::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGemm, {n, n, nb, 1},
                { nullptr, sa, nullptr, sb, nullptr, lsfa, nullptr, lsfb },
                { {1.0f, 0.0f}, nullptr, sc, nullptr, sd }
            };
            cutlass_ws_bytes = GemmOp::get_workspace_size(args);
        }
        // FP16
        {
            using GemmOp = Fp16CutlassGemm;
            using StrideA = typename GemmOp::GemmKernel::StrideA;
            using StrideB = typename GemmOp::GemmKernel::StrideB;
            using StrideC = typename GemmOp::GemmKernel::StrideC;
            using StrideD = typename GemmOp::GemmKernel::StrideD;
            StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {n, nb, 1});
            StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {n, nb, 1});
            StrideC sc = cutlass::make_cute_packed_stride(StrideC{}, {n, n, 1});
            StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {n, n, 1});
            typename GemmOp::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGemm, {n, n, nb, 1},
                { nullptr, sa, nullptr, sb },
                { {1.0f, 0.0f}, nullptr, sc, nullptr, sd }
            };
            size_t ws16 = GemmOp::get_workspace_size(args);
            if (ws16 > cutlass_ws_bytes) cutlass_ws_bytes = ws16;
        }
        // FP8 tensorwise
        {
            using GemmOp = Fp8CutlassGemm;
            using StrideA = typename GemmOp::GemmKernel::StrideA;
            using StrideB = typename GemmOp::GemmKernel::StrideB;
            using StrideC = typename GemmOp::GemmKernel::StrideC;
            using StrideD = typename GemmOp::GemmKernel::StrideD;
            StrideA sa = cutlass::make_cute_packed_stride(StrideA{}, {n, nb, 1});
            StrideB sb = cutlass::make_cute_packed_stride(StrideB{}, {n, nb, 1});
            StrideC sc = cutlass::make_cute_packed_stride(StrideC{}, {n, n, 1});
            StrideD sd = cutlass::make_cute_packed_stride(StrideD{}, {n, n, 1});
            typename GemmOp::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGemm, {n, n, nb, 1},
                { nullptr, sa, nullptr, sb },
                { {1.0f, 0.0f}, nullptr, sc, nullptr, sd }
            };
            size_t ws8 = GemmOp::get_workspace_size(args);
            if (ws8 > cutlass_ws_bytes) cutlass_ws_bytes = ws8;
        }

        cutlass_ws = nullptr;
        if (cutlass_ws_bytes > 0)
            CUDA_CHECK(cudaMalloc(&cutlass_ws, cutlass_ws_bytes));

        // Allocate temps for FP64-max adaptive path
        gemm_temp = nullptr;
        gemm_temp_bytes = 0;
        panel_f32 = nullptr;
        panel_f32_bytes = 0;
        if (need_f64_temps) {
            // GEMM temp: mr x mr float for the largest possible trailing update
            gemm_temp_bytes = sizeof(float) * (size_t)n * n;
            CUDA_CHECK(cudaMalloc(&gemm_temp, gemm_temp_bytes));

            // Panel temp: mr x nb float
            panel_f32_bytes = sizeof(float) * (size_t)n * nb;
            CUDA_CHECK(cudaMalloc(&panel_f32, panel_f32_bytes));
        }
    }

    void free() {
        if (convert_buf) { cudaFree(convert_buf); convert_buf = nullptr; }
        if (scale_A)     { cudaFree(scale_A);     scale_A     = nullptr; }
        if (scale_B)     { cudaFree(scale_B);     scale_B     = nullptr; }
        if (cutlass_ws)  { cudaFree(cutlass_ws);  cutlass_ws  = nullptr; }
        if (gemm_temp)   { cudaFree(gemm_temp);   gemm_temp   = nullptr; }
        if (panel_f32)   { cudaFree(panel_f32);   panel_f32   = nullptr; }
    }
};


// ═══════════════════════════════════════════════════════════════════
//  Cholesky factorization routines
// ═══════════════════════════════════════════════════════════════════

// 1) cuSOLVER Dpotrf (FP64) — takes FP32 matrix, casts to FP64 internally,
//    runs double-precision Cholesky, casts result back to FP32.
void chol_cusolver_fp64(float* d_A32, int n) {
    size_t elems = (size_t)n * n;
    double* d_A64;
    CUDA_CHECK(cudaMalloc(&d_A64, sizeof(double) * elems));
    cast_f32_to_f64(d_A64, d_A32, elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    cusolverDnHandle_t sv;
    CUSOLVER_CHECK(cusolverDnCreate(&sv));
    int lw = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(sv, CUBLAS_FILL_MODE_LOWER, n, d_A64, n, &lw));
    double* dW; int* dI;
    CUDA_CHECK(cudaMalloc(&dW, sizeof(double) * lw));
    CUDA_CHECK(cudaMalloc(&dI, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDpotrf(sv, CUBLAS_FILL_MODE_LOWER, n, d_A64, n, dW, lw, dI));
    CUDA_CHECK(cudaDeviceSynchronize());

    cast_f64_to_f32(d_A32, d_A64, elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_A64));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dI));
    CUSOLVER_CHECK(cusolverDnDestroy(sv));
}

// 2) Templated mixed-precision blocked Cholesky (FP16 or FP8 tensorwise)
template <typename LowT, typename CutlassGemmT, bool DoPerturbDiag = false>
void chol_blocked_mixed(float* d_A, int n, int nb) {
    cusolverDnHandle_t sv; cublasHandle_t bl;
    CUSOLVER_CHECK(cusolverDnCreate(&sv)); CUBLAS_CHECK(cublasCreate(&bl));
    int lw = 0;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(sv, CUBLAS_FILL_MODE_LOWER, nb, d_A, n, &lw));
    float* dW; int* dI;
    CUDA_CHECK(cudaMalloc(&dW, sizeof(float) * std::max(lw, 1)));
    CUDA_CHECK(cudaMalloc(&dI, sizeof(int)));

    float* d_scales;
    CUDA_CHECK(cudaMalloc((void**)&d_scales, sizeof(float)*(n + QUANT_TILE - 1)*(nb + QUANT_TILE - 1)/(QUANT_TILE*QUANT_TILE)));

    // Pre-allocate workspace for the entire factorization
    GemmWorkspace ws;
    ws.allocate(n, nb);

    for (int j = 0; j < n; j += nb) {
        int jb = std::min(nb, n - j);
        float* Ajj = d_A + (size_t)j * n + j;
        CUSOLVER_CHECK(cusolverDnSpotrf(sv, CUBLAS_FILL_MODE_LOWER, jb, Ajj, n, dW, lw, dI));
        if (j + jb >= n) break;
        int mr = n - j - jb;
        float* Ap = d_A + (size_t)j * n + (j + jb);
        float* At = d_A + (size_t)(j + jb) * n + (j + jb);
        float one = 1.f;
        CUBLAS_CHECK(cublasStrsm(bl, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, mr, jb, &one, Ajj, n, Ap, n));

        // Convert panel to low precision and do GEMM update
        LowT* dP = reinterpret_cast<LowT*>(ws.convert_buf);
        launch_convert<LowT>(dP, Ap, d_scales, mr, jb, n, jb);

        if constexpr (DoPerturbDiag) {
            perturb_trailing_diagonal(At, Ap, mr, jb, n,
                9.76562500e-04f*16.0f, 9.76562500e-04f*16.0f);
        }

        run_cutlass_gemm<CutlassGemmT>(mr, mr, jb, dP, dP, At, n,
                                        -1.f, 1.f, ws.cutlass_ws);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    ws.free();
    CUDA_CHECK(cudaFree(dW)); CUDA_CHECK(cudaFree(dI));
    CUBLAS_CHECK(cublasDestroy(bl)); CUSOLVER_CHECK(cusolverDnDestroy(sv));
}

// 3) Adaptive-precision Hybrid Cholesky
struct AdaptiveStats {
    int    step_counts[PREC_COUNT];   // FP8, FP16, TF32, FP32, FP64
    double flops[PREC_COUNT];
    double total_flops;
};


float cuda_event_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// ═══════════════════════════════════════════════════════════════════
//  Adaptive-precision Hybrid Cholesky — templated on HighT
//
//  HighT = double:  A stored in FP64, panel factorized with Dpotrf,
//                   low-prec GEMMs output to temp float buffer then
//                   accumulated into the double trailing matrix.
//                   FP64 GEMM uses cublasDgemm directly.
//
//  HighT = float:   Original behaviour — everything in FP32.
// ═══════════════════════════════════════════════════════════════════

template <typename HighT>
void chol_blocked_adaptive(HighT* d_A, int n, int nb,
                           float c, float eps_prime, bool use_mx,
                           bool verbose = false,
                           AdaptiveStats* out_stats = nullptr) {
    static_assert(std::is_same_v<HighT, float> || std::is_same_v<HighT, double>,
                  "HighT must be float or double");
    constexpr bool is_f64 = std::is_same_v<HighT, double>;

    cublasHandle_t bl;
    CUBLAS_CHECK(cublasCreate(&bl));

    GemmWorkspace ws;
    ws.allocate(n, nb, /*need_f64_temps=*/is_f64);

    // Diagonal buffers — always double for precision selection
    double *d_orig_diag, *d_updated_diag;
    CUDA_CHECK(cudaMalloc(&d_orig_diag,    sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_updated_diag, sizeof(double) * n));

    // Panel factorization info
    int* dI_panel;
    CUDA_CHECK(cudaMalloc(&dI_panel, sizeof(int)));

    float* d_scales;
    CUDA_CHECK(cudaMalloc((void**)&d_scales, sizeof(float)*(n + QUANT_TILE - 1)*(nb + QUANT_TILE - 1)/(QUANT_TILE*QUANT_TILE)));

    // Extract diagonal (always into double)
    if constexpr (is_f64) {
        extract_diagonal_f64(d_orig_diag, d_A, n, n);
    } else {
        // Extract float diagonal, cast to double
        float* d_diag_f32;
        CUDA_CHECK(cudaMalloc(&d_diag_f32, sizeof(float) * n));
        extract_diagonal(d_diag_f32, d_A, n, n);
        cast_f32_to_f64(d_orig_diag, d_diag_f32, n);
        CUDA_CHECK(cudaFree(d_diag_f32));
    }
    CUDA_CHECK(cudaMemcpy(d_updated_diag, d_orig_diag,
                          sizeof(double) * n, cudaMemcpyDeviceToDevice));

    int max_blocks = (n + 255) / 256;
    unsigned int *d_block_masks, *d_result_mask;
    CUDA_CHECK(cudaMalloc(&d_block_masks, sizeof(unsigned int) * max_blocks));
    CUDA_CHECK(cudaMalloc(&d_result_mask, sizeof(unsigned int)));

    int    prec_counts[PREC_COUNT] = {};
    double prec_flops[PREC_COUNT]  = {};

    // One-time cudaFuncSetAttribute for the large-tile fp8 quantization kernel
    {
        constexpr int BLOCK_DIM = 256;
        constexpr int NUM_WARPS = BLOCK_DIM / 32;
        const int padded_rows   = 128 + 1;
        size_t smem = (size_t(padded_rows) * 128 + NUM_WARPS) * sizeof(float);
        CUDA_CHECK(cudaFuncSetAttribute(
            fp32_to_fp8_blockscaled_kernel_opt<BLOCK_DIM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem));
    }
   auto prec_to_beta_delta = [](PrecisionChoice p) -> std::pair<double, double> {
    // δ = β = 16 × smallest positive subnormal of the target precision.
    // This guards against quantization underflow when the trailing
    // diagonal is cast down into the low-precision format.
    switch (p) {
        case PREC_FP8:  return {16.0 * 1.953125e-3,   16.0 * 1.953125e-3};    // E4M3  subnormal = 2^-9
        case PREC_FP16: return {16.0 * 5.9604645e-8,  16.0 * 5.9604645e-8};   // FP16  subnormal = 2^-24
        case PREC_TF32: return {16.0 * 1.4012985e-45, 16.0 * 1.4012985e-45};  // FP32 subnormal range
        case PREC_FP32: return {16.0 * 1.4012985e-45, 16.0 * 1.4012985e-45};  // FP32 subnormal = 2^-149
        case PREC_FP64: return {16.0 * 4.9406565e-324,16.0 * 4.9406565e-324}; // FP64 subnormal = 2^-1074
        default:        return {16.0 * 1.953125e-3,   16.0 * 1.953125e-3};
    }
};



    for (int j = 0; j < n; j += nb) {
        int jb = std::min(nb, n - j);
        HighT* Ajj = d_A + (size_t)j * n + j;
        PrecisionChoice prev_prec = PREC_FP32;
        auto [beta_val, delta_val] = prec_to_beta_delta(prev_prec);  // or prev_prec; see below

        // Panel factorization
        potrf_panel<HighT>(Ajj, jb, n, 99999999.0, 0.0, bl, dI_panel);

        if (j + jb >= n) break;

        int mr = n - j - jb;
        HighT* Ap = d_A + (size_t)j * n + (j + jb);
        HighT* At = d_A + (size_t)(j + jb) * n + (j + jb);

        // Triangular solve
        if constexpr (is_f64) {
            double one = 1.0;
            CUBLAS_CHECK(cublasDtrsm(bl, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, mr, jb,
                                     &one, (double*)Ajj, n, (double*)Ap, n));
        } else {
            float one = 1.f;
            CUBLAS_CHECK(cublasStrsm(bl, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, mr, jb,
                                     &one, (float*)Ajj, n, (float*)Ap, n));
        }

        // Precision selection — always uses double diagonal buffers
        int trailing_start = j + jb;
        PrecisionChoice prec;
        if constexpr (is_f64) {
            prec = select_precision_for_step_f64(
                d_updated_diag, d_orig_diag, (double*)Ap,
                trailing_start, mr, jb, n,
                c, eps_prime, use_mx,
                d_block_masks, d_result_mask);
        } else {
            // For float path, cast Ap temporarily... or just default to FP32.
            // (The float-max path uses the old select_precision_for_step;
            //  but that's a placeholder now.  For correctness just use FP32.)
            prec = PREC_FP32;
        }

        double step_flops = 2.0 * (double)mr * (double)mr * (double)jb;
        prec_counts[prec]++;
        prec_flops[prec] += step_flops;

        if (verbose) {
            printf("  Step j=%d  mr=%d  jb=%d  -> %s  (%.3e FLOPs)\n",
                   j, mr, jb, precision_name(prec), step_flops);
        }

        // For low-precision GEMMs when HighT=double, we need:
        //   1) Cast panel Ap (double, ld=n) → panel_f32 (float, ld=mr)
        //   2) Convert float panel → low-prec
        //   3) GEMM → gemm_temp (float, ld=mr)
        //   4) Accumulate: At -= gemm_temp  (double += float)
        // For FP32 GEMM: cast panel, Sgemm → gemm_temp, accumulate
        // For FP64 GEMM: Dgemm directly on At

        prev_prec = prec;  // for next iteration's diagonal perturbation

        switch (prec) {
            case PREC_FP8: {
                float* panel32 = is_f64 ? ws.panel_f32 : (float*)Ap;
                int ld_panel32 = is_f64 ? mr : n;

                if constexpr (is_f64) {
                    cast_f64_to_f32_matrix(ws.panel_f32, (double*)Ap, mr, jb, mr, n);
                }

                if (use_mx) {
                    constexpr int BLOCK_DIM  = 256;
                    constexpr int SCALE_BM   = 128;
                    constexpr int SCALE_BK   = 128;
                    int nsm = (mr + SCALE_BM - 1) / SCALE_BM;
                    int nsk = (jb + SCALE_BK - 1) / SCALE_BK;

                    const int padded_rows = SCALE_BM + 1;
                    size_t smem_bytes = (size_t(padded_rows) * SCALE_BK + 8) * sizeof(float);

                    fp32_to_fp8_blockscaled_kernel_opt<BLOCK_DIM>
                        <<<dim3(nsm, nsk), BLOCK_DIM, smem_bytes>>>(
                            reinterpret_cast<__nv_fp8_e4m3*>(ws.convert_buf), ws.scale_A,
                            panel32, mr, jb, ld_panel32, jb,
                            SCALE_BM, SCALE_BK, nsm, nsk);
                    CUDA_CHECK(cudaGetLastError());

                    if constexpr (is_f64) {
                        // GEMM → gemm_temp, then accumulate into double At
                        CUDA_CHECK(cudaMemset(ws.gemm_temp, 0, sizeof(float) * (size_t)mr * mr));
                        run_cutlass_gemm_gsfp8(mr, mr, jb,
                            reinterpret_cast<__nv_fp8_e4m3*>(ws.convert_buf),
                            reinterpret_cast<__nv_fp8_e4m3*>(ws.convert_buf),
                            ws.scale_A, ws.scale_A, ws.gemm_temp, mr, 1.f, 0.f,
                            ws.cutlass_ws);
                        accumulate_f32_into_f64_matrix((double*)At, ws.gemm_temp, mr, n, mr, -1.0);
                    } else {
                        run_cutlass_gemm_gsfp8(mr, mr, jb,
                            reinterpret_cast<__nv_fp8_e4m3*>(ws.convert_buf),
                            reinterpret_cast<__nv_fp8_e4m3*>(ws.convert_buf),
                            ws.scale_A, ws.scale_A, (float*)At, n, -1.f, 1.f,
                            ws.cutlass_ws);
                    }
                } else {
                    __nv_fp8_e4m3* p8 = reinterpret_cast<__nv_fp8_e4m3*>(ws.convert_buf);
                    launch_convert<__nv_fp8_e4m3>(p8, panel32, d_scales, mr, jb, ld_panel32, jb);

                    if constexpr (is_f64) {
                        CUDA_CHECK(cudaMemset(ws.gemm_temp, 0, sizeof(float) * (size_t)mr * mr));
                        run_fp8_gemm_scaled(mr, mr, jb, p8, p8,
                            ws.gemm_temp, mr, d_scales, 0.f, ws.cutlass_ws);
                        accumulate_f32_into_f64_matrix((double*)At, ws.gemm_temp, mr, n, mr, -1.0);
                    } else {
                        run_fp8_gemm_scaled(mr, mr, jb, p8, p8,
                            (float*)At, n, d_scales, 1.f, ws.cutlass_ws);
                    }
                }

               
                break;
            }

            case PREC_FP16: {
                float* panel32 = is_f64 ? ws.panel_f32 : (float*)Ap;
                int ld_panel32 = is_f64 ? mr : n;

                if constexpr (is_f64) {
                    cast_f64_to_f32_matrix(ws.panel_f32, (double*)Ap, mr, jb, mr, n);
                }

                __half* p16 = reinterpret_cast<__half*>(ws.convert_buf);
                launch_convert<__half>(p16, panel32, d_scales, mr, jb, ld_panel32, jb);
                CUDA_CHECK(cudaGetLastError());

                if constexpr (is_f64) {
                    CUDA_CHECK(cudaMemset(ws.gemm_temp, 0, sizeof(float) * (size_t)mr * mr));
                    run_fp16_gemm(mr, mr, jb, p16, p16,
                        ws.gemm_temp, mr, 1.f, 0.f, ws.cutlass_ws);
                    accumulate_f32_into_f64_matrix((double*)At, ws.gemm_temp, mr, n, mr, -1.0);
                } else {
                    run_fp16_gemm(mr, mr, jb, p16, p16,
                        (float*)At, n, -1.f, 1.f, ws.cutlass_ws);
                }
                
                //  if constexpr (is_f64) {
                //     // Perturbation for FP8: cast panel to float for perturb kernel
                //     // (panel_f32 already has the cast)
                //     // We need a float view of the diagonal... skip perturbation for f64
                //     // or implement a double version
                //     perturb_trailing_diagonal_f64((double*)At, (double*)Ap, mr, jb, n,
                //         9.76562500e-04f*16.0f, 9.76562500e-04f*16.0f, 0);
                // } else {
                //     perturb_trailing_diagonal((float*)At, (float*)Ap, mr, jb, n,
                //         9.76562500e-04f*16.0f, 9.76562500e-04f*16.0f, 0);
                // }

                break;
            }

            case PREC_TF32: {
                if constexpr (is_f64) {
                    // Cast panel double→float, Sgemm with TF32, accumulate
                    cast_f64_to_f32_matrix(ws.panel_f32, (double*)Ap, mr, jb, mr, n);
                    float neg = -1.f, zero_f = 0.f;
                    CUBLAS_CHECK(cublasSetMathMode(bl, CUBLAS_TF32_TENSOR_OP_MATH));
                    CUBLAS_CHECK(cublasSgemm(bl, CUBLAS_OP_N, CUBLAS_OP_T,
                        mr, mr, jb, &neg, ws.panel_f32, mr, ws.panel_f32, mr,
                        &zero_f, ws.gemm_temp, mr));
                    CUBLAS_CHECK(cublasSetMathMode(bl, CUBLAS_DEFAULT_MATH));
                    accumulate_f32_into_f64_matrix((double*)At, ws.gemm_temp, mr, n, mr, 1.0);
                } else {
                    float neg = -1.f, one_f = 1.f;
                    CUBLAS_CHECK(cublasSetMathMode(bl, CUBLAS_TF32_TENSOR_OP_MATH));
                    CUBLAS_CHECK(cublasSgemm(bl, CUBLAS_OP_N, CUBLAS_OP_T,
                        mr, mr, jb, &neg, (float*)Ap, n, (float*)Ap, n, &one_f, (float*)At, n));
                    CUBLAS_CHECK(cublasSetMathMode(bl, CUBLAS_DEFAULT_MATH));
                }
                break;
            }

            case PREC_FP32: {
                if constexpr (is_f64) {
                    // Cast panel double→float, Sgemm pedantic, accumulate
                    cast_f64_to_f32_matrix(ws.panel_f32, (double*)Ap, mr, jb, mr, n);
                    float neg = -1.f, zero_f = 0.f;
                    CUBLAS_CHECK(cublasSetMathMode(bl, CUBLAS_PEDANTIC_MATH));
                    CUBLAS_CHECK(cublasSgemm(bl, CUBLAS_OP_N, CUBLAS_OP_T,
                        mr, mr, jb, &neg, ws.panel_f32, mr, ws.panel_f32, mr,
                        &zero_f, ws.gemm_temp, mr));
                    CUBLAS_CHECK(cublasSetMathMode(bl, CUBLAS_DEFAULT_MATH));
                    accumulate_f32_into_f64_matrix((double*)At, ws.gemm_temp, mr, n, mr, 1.0);
                } else {
                    float neg = -1.f, one_f = 1.f;
                    CUBLAS_CHECK(cublasSetMathMode(bl, CUBLAS_PEDANTIC_MATH));
                    CUBLAS_CHECK(cublasSgemm(bl, CUBLAS_OP_N, CUBLAS_OP_T,
                        mr, mr, jb, &neg, (float*)Ap, n, (float*)Ap, n, &one_f, (float*)At, n));
                    CUBLAS_CHECK(cublasSetMathMode(bl, CUBLAS_DEFAULT_MATH));
                }
                break;
            }

            case PREC_FP64:
            default: {
                if constexpr (is_f64) {
                    double neg = -1.0, one_d = 1.0;
                    CUBLAS_CHECK(cublasDgemm(bl, CUBLAS_OP_N, CUBLAS_OP_T,
                        mr, mr, jb, &neg, (double*)Ap, n, (double*)Ap, n,
                        &one_d, (double*)At, n));
                } else {
                    // FP64 requested but HighT=float — fall back to FP32 pedantic
                    float neg = -1.f, one_f = 1.f;
                    CUBLAS_CHECK(cublasSetMathMode(bl, CUBLAS_PEDANTIC_MATH));
                    CUBLAS_CHECK(cublasSgemm(bl, CUBLAS_OP_N, CUBLAS_OP_T,
                        mr, mr, jb, &neg, (float*)Ap, n, (float*)Ap, n, &one_f, (float*)At, n));
                    CUBLAS_CHECK(cublasSetMathMode(bl, CUBLAS_DEFAULT_MATH));
                }
                break;
            }
           
        }
        //  auto [beta_val2, delta_val2] = prec_to_beta_delta(prec);
        //      if constexpr (is_f64) {
        //             // Perturbation for FP8: cast panel to float for perturb kernel
        //             // (panel_f32 already has the cast)
        //             // We need a float view of the diagonal... skip perturbation for f64
        //             // or implement a double version
        //             perturb_trailing_diagonal_f64((double*)At, (double*)Ap, mr, jb, n,
        //                 (double)beta_val2, (double)delta_val2, 0);
        //         } else {
        //             perturb_trailing_diagonal((float*)At, (float*)Ap, mr, jb, n,
        //                 (float)beta_val2, (float)delta_val2, 0);
        //         }
    }

    double total_flops = 0;
    for (int p = 0; p < PREC_COUNT; p++) total_flops += prec_flops[p];

    if (verbose) {
        printf("  Adaptive precision summary:\n");
        printf("    FP8:  %3d steps, %12.4e FLOPs (%5.1f%%)\n",
               prec_counts[PREC_FP8],  prec_flops[PREC_FP8],
               total_flops > 0 ? 100.0 * prec_flops[PREC_FP8]  / total_flops : 0.0);
        printf("    FP16: %3d steps, %12.4e FLOPs (%5.1f%%)\n",
               prec_counts[PREC_FP16], prec_flops[PREC_FP16],
               total_flops > 0 ? 100.0 * prec_flops[PREC_FP16] / total_flops : 0.0);
        printf("    TF32: %3d steps, %12.4e FLOPs (%5.1f%%)\n",
               prec_counts[PREC_TF32], prec_flops[PREC_TF32],
               total_flops > 0 ? 100.0 * prec_flops[PREC_TF32] / total_flops : 0.0);
        printf("    FP32: %3d steps, %12.4e FLOPs (%5.1f%%)\n",
               prec_counts[PREC_FP32], prec_flops[PREC_FP32],
               total_flops > 0 ? 100.0 * prec_flops[PREC_FP32] / total_flops : 0.0);
        printf("    FP64: %3d steps, %12.4e FLOPs (%5.1f%%)\n",
               prec_counts[PREC_FP64], prec_flops[PREC_FP64],
               total_flops > 0 ? 100.0 * prec_flops[PREC_FP64] / total_flops : 0.0);
        printf("    Total GEMM FLOPs: %.4e\n", total_flops);
    }

    if (out_stats) {
        for (int p = 0; p < PREC_COUNT; p++) {
            out_stats->step_counts[p] = prec_counts[p];
            out_stats->flops[p]       = prec_flops[p];
        }
        out_stats->total_flops = total_flops;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    ws.free();
    CUDA_CHECK(cudaFree(dI_panel));
    CUDA_CHECK(cudaFree(d_orig_diag)); CUDA_CHECK(cudaFree(d_updated_diag));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_block_masks)); CUDA_CHECK(cudaFree(d_result_mask));
    CUBLAS_CHECK(cublasDestroy(bl));
}

// ═══════════════════════════════════════════════════════════════════
//  Preconditioned Conjugate Gradient  (FP64)
// ═══════════════════════════════════════════════════════════════════

struct PCGResult {
    int iters;
    double rel_residual;
    double time_ms;
    double setup_ms;
};
// ═══════════════════════════════════════════════════════════════════
//  Optimized Preconditioned Conjugate Gradient (FP64)
// ═══════════════════════════════════════════════════════════════════

// ── Device-side scalar helpers ───────────────────────────────────

// alpha = rz_old / pAp;  neg_alpha = -alpha
template<typename T>
__global__ void compute_alpha_kernel(
    const T* __restrict__ rz_old,
    const T* __restrict__ pAp,
    T* __restrict__ alpha,
    T* __restrict__ neg_alpha)
{
    T a = *rz_old / *pAp;
    *alpha = a;
    *neg_alpha = -a;
}

// beta = rz_new / rz_old;  then swap: rz_old = rz_new
template<typename T>
__global__ void compute_beta_and_swap_kernel(
    T* __restrict__ rz_old,
    const T* __restrict__ rz_new,
    T* __restrict__ beta)
{
    T b = *rz_new / *rz_old;
    *beta = b;
    *rz_old = *rz_new;
}

// Fused: x += alpha * p;  r -= alpha * Ap
template<typename T>
__global__ void fused_xr_update_kernel(
    T* __restrict__ x,
    T* __restrict__ r,
    const T* __restrict__ p,
    const T* __restrict__ Ap,
    const T* __restrict__ alpha,
    const T* __restrict__ neg_alpha,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    T a  = *alpha;
    T na = *neg_alpha;
    x[i] += a  * p[i];
    r[i] += na * Ap[i];
}

// Fused: p = z + beta * p  (replaces scal + axpy)
template<typename T>
__global__ void fused_p_update_kernel(
    T* __restrict__ p,
    const T* __restrict__ z,
    const T* __restrict__ beta,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i] = z[i] + (*beta) * p[i];
}

// Check convergence: sets flag[0]=1 if ||r||^2 < (tol*||b||)^2
template<typename T>
__global__ void check_convergence_kernel(
    const T* __restrict__ norm_r_sq,
    T tol_sq_norm_b_sq,
    int* __restrict__ converged)
{
    *converged = (*norm_r_sq < tol_sq_norm_b_sq) ? 1 : 0;
}
PCGResult pcg_solve(int n,
                    const double* d_A,
                    const double* d_b,
                    double* d_x,
                    const double* d_L,
                    int max_iter = 5000,
                    double tol = 1e-10,
                    int check_interval = 1) {

    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));

    // ── Vectors ──────────────────────────────────────────────────
    double *d_r, *d_z, *d_p, *d_Ap;
    CUDA_CHECK(cudaMalloc(&d_r,  sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_z,  sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_p,  sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_Ap, sizeof(double) * n));

    // ── Device scalars (all live on GPU) ─────────────────────────
    double *d_rz_old, *d_rz_new, *d_pAp;
    double *d_alpha, *d_neg_alpha, *d_beta;
    double *d_one, *d_zero;
    int    *d_converged;
    CUDA_CHECK(cudaMalloc(&d_rz_old,    sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rz_new,    sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pAp,       sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha,     sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neg_alpha, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta,      sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_one,       sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_zero,      sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_converged, sizeof(int)));

    { double h = 1.0; CUDA_CHECK(cudaMemcpy(d_one,  &h, sizeof(double), cudaMemcpyHostToDevice)); }
    { double h = 0.0; CUDA_CHECK(cudaMemcpy(d_zero, &h, sizeof(double), cudaMemcpyHostToDevice)); }

    // ── Helper: apply L^{-1} to a vector (in-place) ─────────────
    auto apply_Linv = [&](double* v) {
        if (d_L != nullptr) {
            CUBLAS_CHECK(cublasDtrsv(blas, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                     n, d_L, n, v, 1));
        }
    };

    // ── Helper: apply L^{-T} to a vector (in-place) ─────────────
    auto apply_LTinv = [&](double* v) {
        if (d_L != nullptr) {
            CUBLAS_CHECK(cublasDtrsv(blas, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                     n, d_L, n, v, 1));
        }
    };

    // ── Full preconditioner: z = M^{-1} r  =  L^{-T} L^{-1} r ──
    auto apply_precond = [&](const double* src, double* dst) {
        CUDA_CHECK(cudaMemcpy(dst, src, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        apply_Linv(dst);
        apply_LTinv(dst);
    };

    // ── Compute ||L^{-1} b|| on host (once, before timing) ──────
    // For split PCG, convergence is measured in the preconditioned
    // norm: ||r̂|| / ||b̂||  where r̂ = L^{-1}r, b̂ = L^{-1}b
    double h_norm_bhat = 0;
    {
        double* d_bhat;
        CUDA_CHECK(cudaMalloc(&d_bhat, sizeof(double) * n));
        CUDA_CHECK(cudaMemcpy(d_bhat, d_b, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        apply_Linv(d_bhat);

        CUBLAS_CHECK(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));
        CUBLAS_CHECK(cublasDnrm2(blas, n, d_bhat, 1, &h_norm_bhat));
        CUDA_CHECK(cudaFree(d_bhat));
    }
    if (h_norm_bhat == 0.0) h_norm_bhat = 1.0;

    // Convergence threshold: ||r̂||^2 < (tol * ||b̂||)^2
    // Note: ||r̂||^2 = ||L^{-1}r||^2 = r^T M^{-1} r = r^T z = rz
    double h_tol_sq_nbhat_sq = (tol * h_norm_bhat) * (tol * h_norm_bhat);

    // ── Switch to DEVICE pointer mode for the solve loop ─────────
    CUBLAS_CHECK(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE));

    // r = b  (x = 0, so r = b - Ax = b)
    CUDA_CHECK(cudaMemcpy(d_r, d_b, sizeof(double) * n, cudaMemcpyDeviceToDevice));

    // z = M^{-1} r
    apply_precond(d_r, d_z);

    // p = z
    CUDA_CHECK(cudaMemcpy(d_p, d_z, sizeof(double) * n, cudaMemcpyDeviceToDevice));

    // rz_old = dot(r, z) = ||L^{-1}r||^2 — stays on device
    CUBLAS_CHECK(cublasDdot(blas, n, d_r, 1, d_z, 1, d_rz_old));

    int bl = 256, gr = (n + bl - 1) / bl;

    // ── Timed solve loop ─────────────────────────────────────────
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));

    int iter = 0;
    int h_converged = 0;

    for (iter = 0; iter < max_iter; iter++) {
        // Ap = A * p
        CUBLAS_CHECK(cublasDgemv(blas, CUBLAS_OP_N, n, n,
                                 d_one, d_A, n, d_p, 1, d_zero, d_Ap, 1));

        // pAp = dot(p, Ap)  — on device
        CUBLAS_CHECK(cublasDdot(blas, n, d_p, 1, d_Ap, 1, d_pAp));

        // alpha = rz_old / pAp;  neg_alpha = -alpha  — on device
        compute_alpha_kernel<double><<<1, 1>>>(d_rz_old, d_pAp, d_alpha, d_neg_alpha);

        // Fused: x += alpha*p;  r -= alpha*Ap
        fused_xr_update_kernel<double><<<gr, bl>>>(d_x, d_r, d_p, d_Ap,
                                                    d_alpha, d_neg_alpha, n);

        // z = M^{-1} r
        apply_precond(d_r, d_z);

        // rz_new = dot(r, z) = ||L^{-1} r_{k+1}||^2  — on device
        CUBLAS_CHECK(cublasDdot(blas, n, d_r, 1, d_z, 1, d_rz_new));

        // ── Split-PCG convergence: ||r̂||^2 = rz_new ─────────────
        bool do_check = ((iter + 1) % check_interval == 0) ||
                        (iter + 1 == max_iter);
        if (do_check) {
            // rz_new already holds ||L^{-1}r||^2; compare to threshold
            check_convergence_kernel<double><<<1, 1>>>(d_rz_new, h_tol_sq_nbhat_sq, d_converged);

            CUDA_CHECK(cudaMemcpyAsync(&h_converged, d_converged, sizeof(int),
                                        cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());

            if (h_converged) { iter++; break; }
        }

        // beta = rz_new / rz_old;  rz_old = rz_new  — on device
        compute_beta_and_swap_kernel<double><<<1, 1>>>(d_rz_old, d_rz_new, d_beta);

        // Fused: p = z + beta * p
        fused_p_update_kernel<double><<<gr, bl>>>(d_p, d_z, d_beta, n);
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    // ── Final residual in preconditioned norm ────────────────────
    // rel_res = ||L^{-1} r|| / ||L^{-1} b||
    double rel_res = 1.0;
    {
        double* d_rhat;
        CUDA_CHECK(cudaMalloc(&d_rhat, sizeof(double) * n));
        CUDA_CHECK(cudaMemcpy(d_rhat, d_r, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        apply_Linv(d_rhat);

        CUBLAS_CHECK(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));
        double norm_rhat = 0;
        CUBLAS_CHECK(cublasDnrm2(blas, n, d_rhat, 1, &norm_rhat));
        rel_res = norm_rhat / h_norm_bhat;
        CUDA_CHECK(cudaFree(d_rhat));
    }

    // ── Cleanup ──────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_Ap));
    CUDA_CHECK(cudaFree(d_rz_old));
    CUDA_CHECK(cudaFree(d_rz_new));
    CUDA_CHECK(cudaFree(d_pAp));
    CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_neg_alpha));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_one));
    CUDA_CHECK(cudaFree(d_zero));
    CUDA_CHECK(cudaFree(d_converged));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUBLAS_CHECK(cublasDestroy(blas));

    return {iter, rel_res, (double)ms, 0.0};
}// ═══════════════════════════════════════════════════════════════════
//  Command-line parsing
// ═══════════════════════════════════════════════════════════════════
struct BenchmarkConfig {
    int n       = 4096;
    int nb      = 256;
    double cond = 1e6;
    int max_iter = 10000;
    double tol   = 1e-10;
    std::string csv_path;
    float adapt_c         = 1.0f;
    float adapt_eps_prime = 0.1f;
    std::vector<float> eps_list;
    bool  adapt_use_mx    = false;
    bool  adapt_verbose   = false;
    double shape_p      = 2.0;
    int    num_clusters = 4;
    std::vector<std::string> mtx_paths;   // <── NEW
    int    mtx_max_dim  = 40000;          // <── NEW
};
 
void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --n <int>          Matrix size (default: 4096)\n");
    printf("  --nb <int>         Block size for Cholesky (default: 256)\n");
    printf("  --cond <float>     Condition number (default: 1e6)\n");
    printf("  --max_iter <int>   Max CG iterations (default: 10000)\n");
    printf("  --tol <float>      CG tolerance (default: 1e-10)\n");
    printf("  --csv <path>       Output CSV file (optional)\n");
    printf("  --shape_p <float>  Exponent for poly_decay / power_law (default: 2.0)\n");
    printf("  --num_clusters <int>  Clusters for multi_cluster dist (default: 4)\n");
    printf("  --adapt_c <float>       Adaptive: constant c (default: 1.0)\n");
    printf("  --adapt_eps <float>     Adaptive: single eps_prime (default: 0.1)\n");
    printf("  --adapt_eps_list <csv>  Adaptive: comma-separated eps_prime sweep\n");
    printf("  --adapt_mx              Adaptive: use MXFP8 instead of plain FP8\n");
    printf("  --adapt_verbose         Adaptive: print per-step precision choices\n");
    printf("  --mtx <path>       Matrix Market file (can specify multiple times)\n");
    printf("  --mtx_max_dim <n>  Max dimension for MTX files (default: 40000)\n");
    printf("  --help             Show this help\n");
}
 

 
BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig cfg;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc)
            cfg.n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nb") == 0 && i + 1 < argc)
            cfg.nb = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cond") == 0 && i + 1 < argc)
            cfg.cond = atof(argv[++i]);
        else if (strcmp(argv[i], "--max_iter") == 0 && i + 1 < argc)
            cfg.max_iter = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc)
            cfg.tol = atof(argv[++i]);
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
            cfg.csv_path = argv[++i];
        else if (strcmp(argv[i], "--shape_p") == 0 && i + 1 < argc)
            cfg.shape_p = atof(argv[++i]);
        else if (strcmp(argv[i], "--num_clusters") == 0 && i + 1 < argc)
            cfg.num_clusters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--adapt_c") == 0 && i + 1 < argc)
            cfg.adapt_c = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--adapt_eps") == 0 && i + 1 < argc)
            cfg.adapt_eps_prime = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--adapt_eps_list") == 0 && i + 1 < argc) {
            char* list_str = argv[++i];
            char* tok = strtok(list_str, ",");
            while (tok) {
                cfg.eps_list.push_back((float)atof(tok));
                tok = strtok(nullptr, ",");
            }
        }
        else if (strcmp(argv[i], "--adapt_mx") == 0)
            cfg.adapt_use_mx = true;
        else if (strcmp(argv[i], "--adapt_verbose") == 0)
            cfg.adapt_verbose = true;
        else if (strcmp(argv[i], "--mtx") == 0 && i + 1 < argc)       // <── NEW
            cfg.mtx_paths.push_back(argv[++i]);
        else if (strcmp(argv[i], "--mtx_max_dim") == 0 && i + 1 < argc) // <── NEW
            cfg.mtx_max_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }
    return cfg;
}

// ═══════════════════════════════════════════════════════════════════
//  main — baselines once per (dist), then sweep eps_prime for Adaptive
// ═══════════════════════════════════════════════════════════════════
__global__ void rescale_random_kernel(double* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Map (0, 1] -> [-1, 1]
        data[idx] = (data[idx] * 2.0) - 1.0;
    }
}

void run_benchmark_on_matrix(
    double* d_A64,          // n×n dense SPD matrix (device, not modified)
    int n,
    const char* matrix_name,
    const char* dist_label, // "geometric", "mtx:thermal2", etc.
    BenchmarkConfig& cfg,
    cublasHandle_t cublas_handle,
    cusolverDnHandle_t cusolver_handle,
    cudaStream_t stream,
    FILE* csv_fp)
{
    size_t elems   = (size_t)n * n;
    size_t bytes64 = sizeof(double) * elems;
    size_t bytes32 = sizeof(float)  * elems;
 
    // Allocate working buffers
    double *d_b64, *d_x64, *d_L64;
    float  *d_L32, *d_A32_copy;
    CUDA_CHECK(cudaMalloc(&d_b64, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_x64, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_L64, bytes64));
    CUDA_CHECK(cudaMalloc(&d_L32, bytes32));
    CUDA_CHECK(cudaMalloc(&d_A32_copy, bytes32));
 
    // A32 copy for baseline Cholesky
    cast_f64_to_f32(d_A32_copy, d_A64, elems);
    CUDA_CHECK(cudaDeviceSynchronize());
 
    // b = rnad()
    {
    // 1. Generate uniform doubles in (0, 1] into d_b64
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL)); // Or use a variable seed
    
    CURAND_CHECK(curandGenerateUniformDouble(gen, d_b64, n));

    // 2. Launch custom kernel to rescale to [-1, 1]
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    rescale_random_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_b64, n);

    // 3. Cleanup
    CURAND_CHECK(curandDestroyGenerator(gen));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
 
    // ── Baselines ────────────────────────────────────────────────
    const char* baseline_names[] = { "None", "Dpotrf-FP64", "Mix-FP16" };
 
    printf("\n--- Baselines ---\n");
    printf("%-12s │ %5s │ %12s │ %10s │ %10s │ %10s\n",
           "Precond", "Iters", "Rel Residual", "Setup (ms)", "PCG (ms)", "Total (ms)");
    for (int i = 0; i < 75; i++) printf("─");
    printf("\n");
 
    for (int p = 0; p < 3; p++) {
        CUDA_CHECK(cudaMemset(d_x64, 0, sizeof(double) * n));
 
        double setup_ms = 0.0;
        const double* precond_L64 = nullptr;
 
        if (p > 0) {
            CUDA_CHECK(cudaMemcpy(d_L32, d_A32_copy, bytes32, cudaMemcpyDeviceToDevice));
 
            cudaEvent_t s0, s1;
            CUDA_CHECK(cudaEventCreate(&s0));
            CUDA_CHECK(cudaEventCreate(&s1));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(s0));
 
            switch (p) {
                case 1: chol_cusolver_fp64(d_L32, n); break;
                case 2: chol_blocked_mixed<__half, Fp16CutlassGemm, false>(d_L32, n, cfg.nb); break;
            }
            zero_upper_f32(d_L32, n);
            CUDA_CHECK(cudaDeviceSynchronize());
 
            CUDA_CHECK(cudaEventRecord(s1));
            CUDA_CHECK(cudaEventSynchronize(s1));
            float tmp_ms;
            CUDA_CHECK(cudaEventElapsedTime(&tmp_ms, s0, s1));
            setup_ms = (double)tmp_ms;
            CUDA_CHECK(cudaEventDestroy(s0));
            CUDA_CHECK(cudaEventDestroy(s1));
 
            cast_f32_to_f64(d_L64, d_L32, elems);
            CUDA_CHECK(cudaDeviceSynchronize());
            precond_L64 = d_L64;
        }

 
        PCGResult res = pcg_solve(n, d_A64, d_b64, d_x64,
                                  precond_L64, cfg.max_iter, cfg.tol);
        
                                          //print d_x64
       {
    double norm_A = 0, norm_x = 0, norm_b = 0, norm_r = 0;
    double alpha = 1.0, beta = -1.0;

    // 1. Compute ||A||_F (Frobenius norm)
    // Treating the n x n matrix as one vector of length n*n
    CUBLAS_CHECK(cublasDnrm2(cublas_handle, n * n, d_A64, 1, &norm_A));

    // 2. Compute ||x||_2
    CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_x64, 1, &norm_x));

    // 3. Compute ||b||_2
    CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_b64, 1, &norm_b));

    // 4. Compute the residual vector r = Ax - b
    double* d_res;
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(double)));
    
    // Copy b into the residual buffer first
    CUDA_CHECK(cudaMemcpy(d_res, d_b64, n * sizeof(double), cudaMemcpyDeviceToDevice));
    
    // Perform r = 1.0 * A * x + (-1.0) * r  => r = Ax - b
    CUBLAS_CHECK(cublasDgemv(cublas_handle, CUBLAS_OP_N, n, n, 
                             &alpha, d_A64, n, 
                             d_x64, 1, 
                             &beta, d_res, 1));

    // 5. Compute ||r||_2
    CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_res, 1, &norm_r));

    // 6. Calculate Final Backward Error
    double backward_error = norm_r / (norm_A * norm_x + norm_b);

    printf("\n--- Stability Metrics ---\n");
    printf("  ||Ax - b||:       %.6e\n", norm_r);
    printf("  ||A||_F:           %.6e\n", norm_A);
    printf("  Backward Error:    %.6e\n", backward_error);

    CUDA_CHECK(cudaFree(d_res));
    }

        res.setup_ms = setup_ms;
        double total_ms = res.setup_ms + res.time_ms;
 
        printf("%-12s │ %5d │ %12.2e │ %10.1f │ %10.1f │ %10.1f\n",
               baseline_names[p], res.iters, res.rel_residual,
               res.setup_ms, res.time_ms, total_ms);
        fflush(stdout);
 
        if (csv_fp) {
            fprintf(csv_fp, "%d,%d,%.1e,0,%s,%s,%d,%.6e,%.3f,%.3f,%.3f,0,0,0,0,0,0\n",
                    n, cfg.nb, cfg.cond,
                    dist_label, baseline_names[p], res.iters, res.rel_residual,
                    res.setup_ms, res.time_ms, total_ms);
        }
    }
 
    // ── Adaptive eps_prime sweep ─────────────────────────────────
    printf("\n--- Adaptive eps_prime sweep (%d values) ---\n",
           (int)cfg.eps_list.size());
    printf("%-12s │ %5s │ %12s │ %10s │ %10s │ %10s │%10s │ %8s │ %8s │ %8s | %8s | %8s\n",
           "eps_prime", "Iters", "Rel Residual", "estimated K(inv(L)*A*inv(L))", "Setup (ms)", "PCG (ms)", "Total (ms)",
           "FP8%", "FP16%", "TF32%", "FP32%", "FP64%");
    for (int i = 0; i < 115; i++) printf("─");
    printf("\n");
 
    for (int ei = 0; ei < (int)cfg.eps_list.size(); ei++) {
        float eps = cfg.eps_list[ei];
 
        CUDA_CHECK(cudaMemset(d_x64, 0, sizeof(double) * n));
        CUDA_CHECK(cudaMemcpy(d_L64, d_A64, bytes64, cudaMemcpyDeviceToDevice));
 
        AdaptiveStats adaptive_stats = {};
 
        cudaEvent_t s0, s1;
        CUDA_CHECK(cudaEventCreate(&s0));
        CUDA_CHECK(cudaEventCreate(&s1));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(s0));
 
        chol_blocked_adaptive<double>(d_L64, n, cfg.nb,
                                      cfg.adapt_c, eps,
                                      cfg.adapt_use_mx, cfg.adapt_verbose,
                                      &adaptive_stats);
 
        zero_upper_f64(d_L64, n);
        CUDA_CHECK(cudaDeviceSynchronize());
 
        // NaN check
        {
            int* d_has_nan;
            int h_has_nan = 0;
            CUDA_CHECK(cudaMalloc((void**)&d_has_nan, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_has_nan, 0, sizeof(int)));
            dim3 hsblock(32, 8);
            dim3 hsgrid(std::min((n + 31) / 32, 512), std::min((n + 7) / 8, 512));
            hasNaN_kernel<<<hsgrid, hsblock>>>(d_L64, n, n, d_has_nan);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_has_nan, d_has_nan, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_has_nan));
            if (h_has_nan)
                printf("  WARNING: NaNs detected in Cholesky factor!\n");
        }
 
        CUDA_CHECK(cudaEventRecord(s1));
        CUDA_CHECK(cudaEventSynchronize(s1));
        float tmp_ms;
        CUDA_CHECK(cudaEventElapsedTime(&tmp_ms, s0, s1));
        double setup_ms = (double)tmp_ms;
        CUDA_CHECK(cudaEventDestroy(s0));
        CUDA_CHECK(cudaEventDestroy(s1));

        auto precond_cond = estimate_kappa(
        d_A64, d_L64, n, 100,
        cublas_handle, cusolver_handle);

        CUDA_CHECK(cudaDeviceSynchronize());
 
        PCGResult res = pcg_solve(n, d_A64, d_b64, d_x64,
                                  d_L64, cfg.max_iter, cfg.tol);
        //print d_x64
        {
    double norm_A = 0, norm_x = 0, norm_b = 0, norm_r = 0;
    double alpha = 1.0, beta = -1.0;

    // 1. Compute ||A||_F (Frobenius norm)
    // Treating the n x n matrix as one vector of length n*n
    CUBLAS_CHECK(cublasDnrm2(cublas_handle, n * n, d_A64, 1, &norm_A));

    // 2. Compute ||x||_2
    CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_x64, 1, &norm_x));

    // 3. Compute ||b||_2
    CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_b64, 1, &norm_b));

    // 4. Compute the residual vector r = Ax - b
    double* d_res;
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(double)));
    
    // Copy b into the residual buffer first
    CUDA_CHECK(cudaMemcpy(d_res, d_b64, n * sizeof(double), cudaMemcpyDeviceToDevice));
    
    // Perform r = 1.0 * A * x + (-1.0) * r  => r = Ax - b
    CUBLAS_CHECK(cublasDgemv(cublas_handle, CUBLAS_OP_N, n, n, 
                             &alpha, d_A64, n, 
                             d_x64, 1, 
                             &beta, d_res, 1));

    // 5. Compute ||r||_2
    CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_res, 1, &norm_r));

    // 6. Calculate Final Backward Error
    double backward_error = norm_r / (norm_A * norm_x + norm_b);

    printf("\n--- Stability Metrics ---\n");
    printf("  ||Ax - b||:       %.6e\n", norm_r);
    printf("  ||A||_F:           %.6e\n", norm_A);
    printf("  Backward Error:    %.6e\n", backward_error);

    CUDA_CHECK(cudaFree(d_res));
}
        res.setup_ms = setup_ms;
        double total_ms = res.setup_ms + res.time_ms;
 
        double fp8_pct  = adaptive_stats.total_flops > 0 ? 100.0 * adaptive_stats.flops[PREC_FP8]  / adaptive_stats.total_flops : 0;
        double fp16_pct = adaptive_stats.total_flops > 0 ? 100.0 * adaptive_stats.flops[PREC_FP16] / adaptive_stats.total_flops : 0;
        double tf32_pct = adaptive_stats.total_flops > 0 ? 100.0 * adaptive_stats.flops[PREC_TF32] / adaptive_stats.total_flops : 0;
        double fp32_pct = adaptive_stats.total_flops > 0 ? 100.0 * adaptive_stats.flops[PREC_FP32] / adaptive_stats.total_flops : 0;
        double fp64_pct = adaptive_stats.total_flops > 0 ? 100.0 * adaptive_stats.flops[PREC_FP64] / adaptive_stats.total_flops : 0;
 
        char eps_str[16];
        snprintf(eps_str, sizeof(eps_str), "%.6f", eps);
 
        printf("%-12s │ %5d │ %12.2e │ %10.1f │ %10.1f │ %10.1f │ %10.1f │ %7.1f%% │ %7.1f%% │ %7.1f%% | %7.1f%% | %7.1f%%\n",
               eps_str, res.iters, res.rel_residual, precond_cond, res.setup_ms, res.time_ms,
               total_ms, fp8_pct, fp16_pct, tf32_pct, fp32_pct, fp64_pct);
        fflush(stdout);
 
        if (csv_fp) {
            fprintf(csv_fp, "%d,%d,%.1e,%.6e,%s,%s,%d,%.6e,%.3f,%.3f,%.3f,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e\n",
                    n, cfg.nb, cfg.cond, (double)eps,
                    dist_label, "Adaptive", res.iters, res.rel_residual,
                    res.setup_ms, res.time_ms, total_ms,
                    adaptive_stats.flops[PREC_FP8],
                    adaptive_stats.flops[PREC_FP16],
                    adaptive_stats.flops[PREC_TF32],
                    adaptive_stats.flops[PREC_FP32],
                    adaptive_stats.flops[PREC_FP64],
                    adaptive_stats.total_flops);
        }
    }
 
    // Cleanup
    CUDA_CHECK(cudaFree(d_b64));
    CUDA_CHECK(cudaFree(d_x64));
    CUDA_CHECK(cudaFree(d_L64));
    CUDA_CHECK(cudaFree(d_L32));
    CUDA_CHECK(cudaFree(d_A32_copy));
}
 

int main(int argc, char** argv) {
    BenchmarkConfig cfg = parse_args(argc, argv);
 
    if (cfg.eps_list.empty())
        cfg.eps_list.push_back(cfg.adapt_eps_prime);
    std::sort(cfg.eps_list.begin(), cfg.eps_list.end(), std::greater<float>());
 
    printf("================================================================\n");
    printf("  PCG Benchmark: nb=%d  max_iter=%d  tol=%.1e\n",
           cfg.nb, cfg.max_iter, cfg.tol);
    printf("  Adaptive params: c=%.4f  use_mx=%s\n",
           cfg.adapt_c, cfg.adapt_use_mx ? "yes" : "no");
    printf("  eps_prime sweep (%d values):", (int)cfg.eps_list.size());
    for (float e : cfg.eps_list) printf(" %.6f", e);
    printf("\n");
    if (!cfg.mtx_paths.empty()) {
        printf("  Matrix Market files: %d\n", (int)cfg.mtx_paths.size());
        for (auto& p : cfg.mtx_paths) printf("    %s\n", p.c_str());
    } else {
        printf("  Synthetic: n=%d  cond=%.1e  all %d distributions\n",
               cfg.n, cfg.cond, NUM_DISTS);
    }
    printf("================================================================\n\n");
 
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);
 
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolver_handle, stream));
 
    // One-time cudaFuncSetAttribute for blockscaled fp8 kernel
    {
        constexpr int BLOCK_DIM = 256;
        constexpr int NUM_WARPS = BLOCK_DIM / 32;
        const int padded_rows   = 128 + 1;
        size_t smem = (size_t(padded_rows) * 128 + NUM_WARPS) * sizeof(float);
        CUDA_CHECK(cudaFuncSetAttribute(
            fp32_to_fp8_blockscaled_kernel_opt<BLOCK_DIM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem));
    }
 
    FILE* csv_fp = nullptr;
    if (!cfg.csv_path.empty()) {
        csv_fp = fopen(cfg.csv_path.c_str(), "w");
        if (csv_fp) {
            fprintf(csv_fp, "n,nb,cond,eps_prime,dist,preconditioner,iterations,rel_residual,"
                    "setup_ms,pcg_ms,total_ms,"
                    "gemm_flops_fp8,gemm_flops_fp16,gemm_flops_tf32,gemm_flops_fp32,gemm_flops_fp64,gemm_flops_total\n");
        }
    }
 
    // ──────────────────────────────────────────────────────────────
    //  Branch 1: Matrix Market files
    // ──────────────────────────────────────────────────────────────
    if (!cfg.mtx_paths.empty()) {
        for (int mi = 0; mi < (int)cfg.mtx_paths.size(); mi++) {
            const char* mtx_path = cfg.mtx_paths[mi].c_str();
 
            // Extract matrix name from path for labels
            std::string mtx_name = cfg.mtx_paths[mi];
            size_t slash = mtx_name.rfind('/');
            if (slash != std::string::npos) mtx_name = mtx_name.substr(slash + 1);
            size_t dot = mtx_name.rfind('.');
            if (dot != std::string::npos) mtx_name = mtx_name.substr(0, dot);
 
            printf("════════════════════════════════════════════════════════════════\n");
            printf("  Matrix Market: %s\n", mtx_name.c_str());
            printf("════════════════════════════════════════════════════════════════\n");
 
            double* d_A64 = nullptr;
            int n = 0;
            int rc = load_mtx_to_dense_f64(mtx_path, &d_A64, &n, cfg.mtx_max_dim);
            if (rc != 0) {
                printf("  SKIPPING (load failed)\n\n");
                continue;
            }
 
            // Override cfg.n for this matrix (used by workspace allocations inside)
            int saved_n = cfg.n;
            cfg.n = n;
 
            // For CSV, use "mtx:<name>" as the distribution label
            char dist_label[256];
            snprintf(dist_label, sizeof(dist_label), "mtx:%s", mtx_name.c_str());
 
            // Set cond to 0 to indicate "unknown" for MTX matrices
            double saved_cond = cfg.cond;
            cfg.cond = 0.0;
 
            run_benchmark_on_matrix(d_A64, n, mtx_name.c_str(), dist_label,
                                    cfg, cublas_handle, cusolver_handle,
                                    stream, csv_fp);
 
            cfg.n = saved_n;
            cfg.cond = saved_cond;
 
            CUDA_CHECK(cudaFree(d_A64));
            printf("\n");
        }
    }
 
    // ──────────────────────────────────────────────────────────────
    //  Branch 2: Synthetic matrices (only if no MTX files given,
    //            or always — change the condition below if you want both)
    // ──────────────────────────────────────────────────────────────
    if (cfg.mtx_paths.empty()) {
        int n = cfg.n;
        size_t elems   = (size_t)n * n;
        size_t bytes64 = sizeof(double) * elems;
 
        double* d_A64;
        CUDA_CHECK(cudaMalloc(&d_A64, bytes64));
 
        for (int di = 0; di < NUM_DISTS; di++) {
            eigenvalue_dist_t dist = ALL_DISTS[di];
            const char* dname = dist_name(dist);
 
            char dist_label[128];
            if (dist == DIST_POLY_DECAY || dist == DIST_POWER_LAW)
                snprintf(dist_label, sizeof(dist_label), "%s(p=%.1f)", dname, cfg.shape_p);
            else if (dist == DIST_MULTI_CLUSTER)
                snprintf(dist_label, sizeof(dist_label), "%s(k=%d)", dname, cfg.num_clusters);
            else
                snprintf(dist_label, sizeof(dist_label), "%s", dname);
 
            printf("════════════════════════════════════════════════════════════════\n");
            printf("  Distribution: %s\n", dist_label);
            printf("════════════════════════════════════════════════════════════════\n");
 
            printf("Generating %d x %d SPD matrix (cond=%.1e, dist=%s) ... ",
                   n, n, cfg.cond, dist_label);
            fflush(stdout);
            generate_spd(d_A64, n, n, cfg.cond, dist, cublas_handle,
                         cusolver_handle, stream, cfg.shape_p, cfg.num_clusters);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            printf("done.\n");
 
            run_benchmark_on_matrix(d_A64, n, dist_label, dname,
                                    cfg, cublas_handle, cusolver_handle,
                                    stream, csv_fp);
            printf("\n");
        }
 
        CUDA_CHECK(cudaFree(d_A64));
    }
 
    if (csv_fp) {
        fclose(csv_fp);
        printf("Results written to: %s\n", cfg.csv_path.c_str());
    }
 
    printf("PCG tolerance: %.0e, max iterations: %d\n", cfg.tol, cfg.max_iter);
 
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
 
    return 0;
}
 