/*
 * pcg_benchmark.cu
 *
 * Preconditioned Conjugate Gradient in FP64, comparing different
 * single-precision Cholesky factorizations as preconditioners.
 *
 * Preconditioners:
 *   0) None (plain CG)
 *   1) cuSOLVER Dpotrf (FP64) cast back to FP32
 *   2) Mixed FP16 Cholesky  (cuBLASLt FP16 GEMM, FP32 accum/output)
 *   3) Adaptive-precision Hybrid Cholesky (per-step precision selection)
 *      - Low-precision path uses cuBLASLt FP8 with 1-D block scaling
 *        (128 along K, 1 along M/N) → row-scaled MX-style
 *
 * CG runs in FP32 or FP64 depending on path.
 *
 * Test matrix:  A = Q diag(λ) Q^T   with eigenvalues from 1 to 1/κ
 * under various distributions.  Built as FP64 then cast to FP32.
 *
 * Build (sm_90a, CUDA 12.9+ required for FP8 1-D block scaling):
 *   nvcc -std=c++17 -O3 -arch=sm_90a -Xptxas --allow-expensive-optimizations=true --expt-relaxed-constexpr pcg_bench.cu mmio.c -lcusolver -lcublas -lcublasLt -lcurand -o pcg_bench
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
#include <unordered_map>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <curand.h>

extern "C" {
#include "mmio.h"
}

#define QUANT_TILE 32

#define COUNT_UNDERFLOW

// ─── Error macros ────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                      \
    cudaError_t e = (call);                                        \
    if (e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(e));        \
        exit(1);                                                   \
    }                                                              \
} while (0)

#define CUSOLVER_CHECK(call) do {                                  \
    cusolverStatus_t s = (call);                                   \
    if (s != CUSOLVER_STATUS_SUCCESS) {                            \
        fprintf(stderr, "cuSOLVER %s:%d: %d\n",                    \
                __FILE__, __LINE__, (int)s); exit(1);              \
    }                                                              \
} while (0)

#define CUBLAS_CHECK(call) do {                                    \
    cublasStatus_t s = (call);                                     \
    if (s != CUBLAS_STATUS_SUCCESS) {                              \
        fprintf(stderr, "cuBLAS %s:%d: %d\n",                      \
                __FILE__, __LINE__, (int)s); exit(1);              \
    }                                                              \
} while (0)

#define CURAND_CHECK(call) do {                                    \
    curandStatus_t s = (call);                                     \
    if (s != CURAND_STATUS_SUCCESS) {                              \
        fprintf(stderr, "cuRAND %s:%d: %d\n",                      \
                __FILE__, __LINE__, (int)s); exit(1);              \
    }                                                              \
} while (0)

// ═══════════════════════════════════════════════════════════════════
//  Scale-mode enum names changed during cuBLAS 12.9 development.
//  The release name in 12.9+ headers is CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F
//  (the "VEC128" refers to one FP32 scale per 128-element vector along K).
//  For older preview headers or different toolkit minor versions, the
//  spelling may be CUBLASLT_MATMUL_MATRIX_SCALE_BLK128_1D_32F or
//  CUBLASLT_MATMUL_MATRIX_SCALE_BLK128_OUTER_VEC_32F.  Add fallbacks.
//  Check with:  grep MATMUL_MATRIX_SCALE_ /usr/local/cuda/include/cublasLt.h
// ═══════════════════════════════════════════════════════════════════
#if !defined(CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F)
  #if defined(CUBLASLT_MATMUL_MATRIX_SCALE_BLK128_1D_32F)
    #define CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F \
            CUBLASLT_MATMUL_MATRIX_SCALE_BLK128_1D_32F
  #elif defined(CUBLASLT_MATMUL_MATRIX_SCALE_BLK128_OUTER_VEC_32F)
    #define CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F \
            CUBLASLT_MATMUL_MATRIX_SCALE_BLK128_OUTER_VEC_32F
  #endif
#endif

// ─── Small kernel: random normal init for Lanczos starting vector ───────────
__global__ void init_lanczos_start_kernel(double* __restrict__ v, int n, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned h = (unsigned)i * 2654435761u + seed;
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u; h ^= h >> 16;
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

static inline double compute_true_residual(
        const double* d_A, const double* d_b, const double* d_x,
        int n, cublasHandle_t blas)
{
    double* d_r;
    cudaMalloc(&d_r, sizeof(double) * n);
    cudaMemcpy(d_r, d_b, sizeof(double) * n, cudaMemcpyDeviceToDevice);

    cublasPointerMode_t old_mode;
    cublasGetPointerMode(blas, &old_mode);
    cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST);

    double neg = -1.0, one = 1.0;
    cublasDgemv(blas, CUBLAS_OP_N, n, n, &neg, d_A, n, d_x, 1, &one, d_r, 1);

    double norm_r = 0.0, norm_b = 0.0;
    cublasDnrm2(blas, n, d_r, 1, &norm_r);
    cublasDnrm2(blas, n, d_b, 1, &norm_b);

    cublasSetPointerMode(blas, old_mode);
    cudaFree(d_r);

    return (norm_b > 0) ? norm_r / norm_b : norm_r;
}

// ═══════════════════════════════════════════════════════════════════
//  cuBLASLt persistent context & GEMM launchers
// ═══════════════════════════════════════════════════════════════════

class CublasLtContext {
public:
    void init(size_t workspace_bytes = 32 * 1024 * 1024) {
        CUBLAS_CHECK(cublasLtCreate(&handle_));
        workspace_bytes_ = workspace_bytes;
        CUDA_CHECK(cudaMalloc(&workspace_, workspace_bytes_));
    }

    void destroy() {
        if (workspace_) { cudaFree(workspace_); workspace_ = nullptr; }
        if (handle_)    { cublasLtDestroy(handle_); handle_ = nullptr; }
    }

    cublasLtHandle_t handle() const { return handle_; }
    void*  workspace()        const { return workspace_; }
    size_t workspace_bytes()  const { return workspace_bytes_; }

private:
    cublasLtHandle_t handle_       = nullptr;
    void*            workspace_    = nullptr;
    size_t           workspace_bytes_ = 0;
};

struct LtShapeKey {
    int M, N, K;
    int lda, ldb, ldc;
    bool operator==(const LtShapeKey& o) const {
        return M == o.M && N == o.N && K == o.K
            && lda == o.lda && ldb == o.ldb && ldc == o.ldc;
    }
};
struct LtShapeKeyHash {
    size_t operator()(const LtShapeKey& k) const noexcept {
        size_t h = 1469598103934665603ull;            // FNV-1a
        auto mix = [&](size_t v){ h ^= v; h *= 1099511628211ull; };
        mix((size_t)k.M);   mix((size_t)k.N);   mix((size_t)k.K);
        mix((size_t)k.lda); mix((size_t)k.ldb); mix((size_t)k.ldc);
        return h;
    }
};

struct LtPlan {
    cublasLtMatmulDesc_t           op_desc   = nullptr;
    cublasLtMatrixLayout_t         A_layout  = nullptr;
    cublasLtMatrixLayout_t         B_layout  = nullptr;
    cublasLtMatrixLayout_t         C_layout  = nullptr;
    cublasLtMatrixLayout_t         D_layout  = nullptr;
    cublasLtMatmulAlgo_t           algo      {};
    bool                           valid     = false;
};

inline void destroy_lt_plan(LtPlan& p) {
    if (p.D_layout)  { cublasLtMatrixLayoutDestroy(p.D_layout);  p.D_layout = nullptr; }
    if (p.C_layout)  { cublasLtMatrixLayoutDestroy(p.C_layout);  p.C_layout = nullptr; }
    if (p.B_layout)  { cublasLtMatrixLayoutDestroy(p.B_layout);  p.B_layout = nullptr; }
    if (p.A_layout)  { cublasLtMatrixLayoutDestroy(p.A_layout);  p.A_layout = nullptr; }
    if (p.op_desc)   { cublasLtMatmulDescDestroy(p.op_desc);     p.op_desc  = nullptr; }
    p.valid = false;
}

// ═══════════════════════════════════════════════════════════════════
//  FP16 launcher  (FP16 input row-major panel, FP32 output col-major)
//  TN configuration matching the symmetric L*L^T trailing update.
// ═══════════════════════════════════════════════════════════════════

class Fp16GemmLt {
public:
    void set_context(CublasLtContext* ctx) { ctx_ = ctx; }

    void run(int M, int N, int K,
             const __half* A, int lda,
             const __half* B, int ldb,
             float* C, int ldc,
             float alpha, float beta,
             cudaStream_t stream)
    {
        LtPlan& plan = get_or_build_plan(M, N, K, lda, ldb, ldc);
        CUBLAS_CHECK(cublasLtMatmul(
            ctx_->handle(),
            plan.op_desc,
            &alpha,
            A, plan.A_layout,
            B, plan.B_layout,
            &beta,
            C, plan.C_layout,
            C, plan.D_layout,
            &plan.algo,
            ctx_->workspace(),
            ctx_->workspace_bytes(),
            stream));
    }

    void clear_cache() {
        for (auto& kv : cache_) destroy_lt_plan(kv.second);
        cache_.clear();
    }

    ~Fp16GemmLt() { clear_cache(); }

private:
    LtPlan& get_or_build_plan(int M, int N, int K,
                              int lda, int ldb, int ldc)
    {
        LtShapeKey key{M, N, K, lda, ldb, ldc};   // was {M, N, K}
        auto it = cache_.find(key);
        if (it != cache_.end() && it->second.valid) return it->second;

        LtPlan plan;

        CUBLAS_CHECK(cublasLtMatmulDescCreate(
            &plan.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

        cublasOperation_t opA = CUBLAS_OP_T;
        cublasOperation_t opB = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &plan.A_layout, CUDA_R_16F, K, M, lda));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &plan.B_layout, CUDA_R_16F, K, N, ldb));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &plan.C_layout, CUDA_R_32F, M, N, ldc));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &plan.D_layout, CUDA_R_32F, M, N, ldc));

        cublasLtMatmulPreference_t pref;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        size_t ws = ctx_->workspace_bytes();
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws, sizeof(ws)));

        cublasLtMatmulHeuristicResult_t heur{};
        int returned = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
            ctx_->handle(), plan.op_desc,
            plan.A_layout, plan.B_layout, plan.C_layout, plan.D_layout,
            pref, 1, &heur, &returned));
        cublasLtMatmulPreferenceDestroy(pref);

        if (returned == 0) {
            fprintf(stderr,
                "Fp16GemmLt: no cuBLASLt algorithm for M=%d N=%d K=%d\n",
                M, N, K);
            std::exit(1);
        }
        plan.algo  = heur.algo;
        plan.valid = true;

        auto [ins_it, _] = cache_.emplace(key, std::move(plan));
        return ins_it->second;
    }

    CublasLtContext*                                              ctx_ = nullptr;
    std::unordered_map<LtShapeKey, LtPlan, LtShapeKeyHash>        cache_;
};

// ═══════════════════════════════════════════════════════════════════
//  FP8 tensorwide-scaled launcher (single FP32 scalar per matrix).
//
//  Why tensorwide and not VEC128 1-D block scaling?
//    The VEC128 path requires a specific scale-array layout that
//    is poorly documented and easy to get wrong, leading to silent
//    NaN production.  Tensorwide scaling is the original (cuBLAS 12.0+)
//    FP8 path documented in NVIDIA's LtFp8Matmul sample and is widely
//    used in production code (e.g. PyTorch's torch._scaled_mm default).
//
//  Inputs:
//    A_fp8 : __nv_fp8_e4m3, ROW-MAJOR M × K, ld_row = K
//    B_fp8 : __nv_fp8_e4m3, COL-MAJOR view K × N, ld_col = K
//            (For symmetric L*L^T, same physical buffer as A.)
//    a_scale: device pointer to FP32 vector of length M (one scale per row of A)
//    b_scale: device pointer to FP32 vector of length N (one scale per col of B)
//             For symmetric L*L^T, same buffer as a_scale (since A's rows
//             correspond to B's columns when B is the transpose view).
//
//  Output: __nv_bfloat16, col-major M × N, ld = ldd.
//
//  Uses CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F: per-row scaling for A,
//  per-column scaling for B.  This matches PyTorch's "rowwise" scaled_mm
//  path which uses scale_a (M, 1) and scale_b (1, N).
// ═══════════════════════════════════════════════════════════════════

class Fp8BlockScaledGemmLt {
public:
    void set_context(CublasLtContext* ctx) { ctx_ = ctx; }

    void run(int M, int N, int K,
             const __nv_fp8_e4m3* A_fp8,
             const __nv_fp8_e4m3* B_fp8,
             const float* a_scale,
             const float* b_scale,
             __nv_bfloat16* D, int ldd,
             float alpha, float beta,
             cudaStream_t stream)
    {
        LtPlan& plan = get_or_build_plan(M, N, K, ldd, a_scale, b_scale);

        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            plan.op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &a_scale, sizeof(a_scale)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            plan.op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &b_scale, sizeof(b_scale)));

        CUBLAS_CHECK(cublasLtMatmul(
            ctx_->handle(),
            plan.op_desc,
            &alpha,
            A_fp8, plan.A_layout,
            B_fp8, plan.B_layout,
            &beta,
            D, plan.C_layout,
            D, plan.D_layout,
            &plan.algo,
            ctx_->workspace(),
            ctx_->workspace_bytes(),
            stream));
    }

    void clear_cache() {
        for (auto& kv : cache_) destroy_lt_plan(kv.second);
        cache_.clear();
    }

    ~Fp8BlockScaledGemmLt() { clear_cache(); }

private:
    LtPlan& get_or_build_plan(int M, int N, int K, int ldd,
                              const float* a_scale, const float* b_scale)
    {
        LtShapeKey key{M, N, K, K, K, ldd};       // was {M, N, K}
        auto it = cache_.find(key);
        if (it != cache_.end() && it->second.valid) return it->second;

        LtPlan plan;

        CUBLAS_CHECK(cublasLtMatmulDescCreate(
            &plan.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

        // TN configuration (Hopper FP8 requirement)
        cublasOperation_t opA = CUBLAS_OP_T;
        cublasOperation_t opB = CUBLAS_OP_N;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

        // Per-row scaling for A, per-column scaling for B.
        // PyTorch's "rowwise" path uses this same enum.
        // (Without explicitly setting this attribute, cuBLAS defaults to
        //  SCALAR_32F = tensorwide.  Setting it explicitly is required
        //  to engage the per-row mode.)
        cublasLtMatmulMatrixScale_t scale_mode =
            CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            plan.op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
            &scale_mode, sizeof(scale_mode)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            plan.op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
            &scale_mode, sizeof(scale_mode)));

        // Scale pointers (required even at heuristic time when FP8 is used).
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            plan.op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &a_scale, sizeof(a_scale)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            plan.op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &b_scale, sizeof(b_scale)));

        // Fast-accum is permitted with rowwise (OUTER_VEC) scaling.
        // (PyTorch disallows fast_accum only for 128-element 1D and
        //  128x128 2D blockwise scaling — rowwise is fine.)
        int8_t fast_accum = 0;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            plan.op_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
            &fast_accum, sizeof(fast_accum)));

        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &plan.A_layout, CUDA_R_8F_E4M3, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &plan.B_layout, CUDA_R_8F_E4M3, K, N, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &plan.C_layout, CUDA_R_16BF, M, N, ldd));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
            &plan.D_layout, CUDA_R_16BF, M, N, ldd));

        cublasLtMatmulPreference_t pref;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        size_t ws = ctx_->workspace_bytes();
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws, sizeof(ws)));

        cublasLtMatmulHeuristicResult_t heur{};
        int returned = 0;
        cublasStatus_t hstat = cublasLtMatmulAlgoGetHeuristic(
            ctx_->handle(), plan.op_desc,
            plan.A_layout, plan.B_layout, plan.C_layout, plan.D_layout,
            pref, 1, &heur, &returned);
        cublasLtMatmulPreferenceDestroy(pref);

        if (hstat != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                "Fp8 tensorwide GEMM: heuristic returned %d for M=%d N=%d K=%d\n",
                (int)hstat, M, N, K);
            std::exit(1);
        }
        if (returned == 0) {
            fprintf(stderr,
                "Fp8 tensorwide GEMM: no algorithm for M=%d N=%d K=%d\n",
                M, N, K);
            std::exit(1);
        }
        plan.algo  = heur.algo;
        plan.valid = true;

        auto [ins_it, _] = cache_.emplace(key, std::move(plan));
        return ins_it->second;
    }

    CublasLtContext*                                              ctx_ = nullptr;
    std::unordered_map<LtShapeKey, LtPlan, LtShapeKeyHash>        cache_;
};

// ═══════════════════════════════════════════════════════════════════
//  Global cuBLASLt context & launchers
// ═══════════════════════════════════════════════════════════════════
static CublasLtContext       g_lt_ctx;
static Fp16GemmLt            g_fp16_gemm;
static Fp8BlockScaledGemmLt  g_fp8_gemm;

// ═══════════════════════════════════════════════════════════════════
//  FP8 row-wise quantization kernel
//
//  Reads FP32 col-major panel (rows × cols, ld = ld_src)
//  Writes:
//    dst    : FP8 row-major, ld = cols (= K)
//    scales : FP32 vector of length `rows`.
//             scales[m] = amax(row m of source) / 448
//
//  One warp per row.  Each warp does:
//    Pass 1: lanes stride across the row to find amax, then warp-reduce.
//    Pass 2: lane 0 writes the scale; all lanes quantize their slice.
//
//  This matches cuBLAS's OUTER_VEC_32F expectation: flat FP32 vector,
//  length M for the A operand (and length N for B — for the symmetric
//  L*L^T case the same buffer serves both since rows of A correspond
//  to cols of B^T).
// ═══════════════════════════════════════════════════════════════════

namespace fp8bs1d {

constexpr float FP8_MAX_V    = 448.0f;
// Quantization target: rather than mapping each row's amax to FP8_MAX_V (448),
// we map it to sqrt(FP8_MAX_V) ≈ 21.17.  This leaves ~21× headroom per operand,
// so the products in the GEMM accumulator land near 21*21 ≈ 441 rather than
// 448*448 ≈ 2e5 — well within FP16's range when FAST_ACCUM is on.
// Costs ~4.4 bits of dynamic range for peak values, prevents overflow NaN.
constexpr float FP8_TARGET   = 11.166010488516726f;  // sqrtf(448) - 10
constexpr float AMAX_FLOOR   = 9.76562500e-04f * 16.0f;

__device__ __forceinline__ float warp_reduce_max_local(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  8));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  4));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  2));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v,  1));
    return v;
}

#ifdef COUNT_UNDERFLOW
__device__ __forceinline__ int warp_reduce_sum_local(int v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v,  8);
    v += __shfl_xor_sync(0xffffffff, v,  4);
    v += __shfl_xor_sync(0xffffffff, v,  2);
    v += __shfl_xor_sync(0xffffffff, v,  1);
    return v;
}
#endif

// One warp per row.  Grid: (ceil(rows / WARPS_PER_CTA), 1).
template<int WARPS_PER_CTA>
__global__ void fp32_to_fp8_rowwise_kernel(
    __nv_fp8_e4m3* __restrict__ dst,     // row-major, ld = cols
    float*         __restrict__ scales,  // vector of length rows
    const float*   __restrict__ src,     // col-major, ld = ld_src
    int rows, int cols, int ld_src
#ifdef COUNT_UNDERFLOW
    // Accumulator indexed by GLOBAL row of L (caller passes the array
    // pre-offset to this call's first row); persists across calls so
    // it tallies underflows over the whole matrix's factorization, not
    // just this one panel.
    , int* __restrict__ row_underflow_counts
#endif
    )
{
    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * WARPS_PER_CTA + wid;
    if (row >= rows) return;

    // Pass 1: amax over the whole row
    float local_max = 0.f;
    for (int k = lane; k < cols; k += 32) {
        float v = src[(size_t)k * ld_src + row];
        local_max = fmaxf(local_max, fabsf(v));
    }
    local_max = warp_reduce_max_local(local_max);

    const float scale     = (local_max > AMAX_FLOOR) ? (local_max / FP8_TARGET) : 1.f;
    const float inv_scale = 1.f / scale;

    if (lane == 0) scales[row] = scale;

#ifdef COUNT_UNDERFLOW
    int local_underflow = 0;
#endif

    // Pass 2: quantize and write row-major
    for (int k = lane; k < cols; k += 32) {
        float orig = src[(size_t)k * ld_src + row];
        float v = orig * inv_scale;
        // Clamp to FP8 representable range (±448).  In normal operation
        // the largest value lands near ±FP8_TARGET ≈ ±21; the clamp only
        // catches outliers from the AMAX_FLOOR branch or numerical edge cases.
        v = fminf(fmaxf(v, -FP8_MAX_V), FP8_MAX_V);
        __nv_fp8_e4m3 q = __nv_fp8_e4m3(v);
#ifdef COUNT_UNDERFLOW
        if (orig != 0.f && float(q) == 0.f) local_underflow++;
#endif
        dst[(size_t)row * cols + k] = q;
    }

#ifdef COUNT_UNDERFLOW
    int row_underflow = warp_reduce_sum_local(local_underflow);
    if (lane == 0 && row_underflow > 0)
        atomicAdd(&row_underflow_counts[row], row_underflow);
#endif
}

inline void launch(
    __nv_fp8_e4m3* dst,
    float*         scales,        // FP32 vector of length `rows`
    const float*   src,
    int rows, int cols, int ld_src,
    cudaStream_t   stream
#ifdef COUNT_UNDERFLOW
    , int* row_underflow_counts   // device array, pre-offset to global row 0 of this call
#endif
    )
{
    constexpr int WARPS_PER_CTA = 4;
    const int grid_x = (rows + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    fp32_to_fp8_rowwise_kernel<WARPS_PER_CTA><<<grid_x, WARPS_PER_CTA * 32, 0, stream>>>(
        dst, scales, src, rows, cols, ld_src
#ifdef COUNT_UNDERFLOW
        , row_underflow_counts
#endif
        );
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace fp8bs1d

// ═══════════════════════════════════════════════════════════════════
//  Lanczos extreme-eigenvalue estimation
// ═══════════════════════════════════════════════════════════════════
#ifdef COND_APPROX

static inline void lanczos_extreme_eigs(
        const double* d_A, const double* d_L, int n, int k_max,
        double& lambda_min, double& lambda_max,
        cublasHandle_t blas, cusolverDnHandle_t cusolver)
{
    if (k_max > n) k_max = n;

    double *d_V, *d_w, *d_tmp;
    cudaMalloc(&d_V,  sizeof(double) * (size_t)n * k_max);
    cudaMalloc(&d_w,  sizeof(double) * n);
    cudaMalloc(&d_tmp,sizeof(double) * n);

    std::vector<double> alpha(k_max, 0.0), beta(k_max, 0.0);

    cublasPointerMode_t old_mode;
    cublasGetPointerMode(blas, &old_mode);
    cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST);

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

        if (d_L == nullptr) {
            double one = 1.0, zero = 0.0;
            cublasDgemv(blas, CUBLAS_OP_N, n, n, &one, d_A, n, vj, 1, &zero, d_w, 1);
        } else {
            cudaMemcpy(d_w, vj, sizeof(double) * n, cudaMemcpyDeviceToDevice);
            cublasDtrsv(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                        CUBLAS_DIAG_NON_UNIT, n, d_L, n, d_w, 1);
            double one = 1.0, zero = 0.0;
            cublasDgemv(blas, CUBLAS_OP_N, n, n, &one, d_A, n, d_w, 1, &zero, d_tmp, 1);
            cudaMemcpy(d_w, d_tmp, sizeof(double) * n, cudaMemcpyDeviceToDevice);
            cublasDtrsv(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                        CUBLAS_DIAG_NON_UNIT, n, d_L, n, d_w, 1);
        }

        cublasDdot(blas, n, vj, 1, d_w, 1, &alpha[j]);

        double na = -alpha[j];
        cublasDaxpy(blas, n, &na, vj, 1, d_w, 1);
        if (j > 0) {
            double nb = -beta[j - 1];
            double* vjm1 = d_V + (size_t)(j - 1) * n;
            cublasDaxpy(blas, n, &nb, vjm1, 1, d_w, 1);
        }

        for (int i = 0; i <= j; ++i) {
            double* vi = d_V + (size_t)i * n;
            double dot;
            cublasDdot(blas, n, vi, 1, d_w, 1, &dot);
            double ndot = -dot;
            cublasDaxpy(blas, n, &ndot, vi, 1, d_w, 1);
        }

        cublasDnrm2(blas, n, d_w, 1, &beta[j]);

        if (beta[j] < 1e-14) { k_used = j + 1; break; }

        if (j + 1 < k_max) {
            double inv = 1.0 / beta[j];
            cudaMemcpy(vjp1, d_w, sizeof(double) * n, cudaMemcpyDeviceToDevice);
            cublasDscal(blas, n, &inv, vjp1, 1);
        }
    }
    cublasSetPointerMode(blas, old_mode);

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

    lambda_min = evals.front();
    lambda_max = evals.back();

    cudaFree(d_T); cudaFree(d_evals); cudaFree(d_work); cudaFree(d_info);
    cudaFree(d_V); cudaFree(d_w); cudaFree(d_tmp);
}

static inline double estimate_kappa(
        const double* d_A, const double* d_L, int n, int k_max,
        cublasHandle_t blas, cusolverDnHandle_t cusolver)
{
    double lmin, lmax;
    lanczos_extreme_eigs(d_A, d_L, n, k_max, lmin, lmax, blas, cusolver);
    if (lmin <= 0) return -1.0;
    return lmax / lmin;
}

#endif  // COND_APPROX

// ═══════════════════════════════════════════════════════════════════
//  ExtraMetrics
// ═══════════════════════════════════════════════════════════════════
struct ExtraMetrics {
    double forward_err;
    double true_resid;
    double time_per_iter_ms;
    double kappa_A_est;
    double kappa_MinvA_est;
    int    nan_flag;
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
    DIST_GEOMETRIC, DIST_RANDOM,
    DIST_ARITHMETIC, DIST_MULTI_CLUSTER
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

// ═══════════════════════════════════════════════════════════════════
//  Precision enum
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

// Mantissa (stored-fraction) bits per precision tier, for the Figure-5
// per-panel staircase.  FP8 is E4M3 (3 mantissa bits); TF32 carries a
// 10-bit mantissa like FP16; bf16 (7) is not directly selectable here.
static int mantissa_bits_for_prec(PrecisionChoice p) {
    switch (p) {
        case PREC_FP8:  return 3;
        case PREC_FP16: return 10;
        case PREC_TF32: return 10;
        case PREC_FP32: return 23;
        case PREC_FP64: return 52;
        default:        return 52;
    }
}

// Monotonically increasing identifier assigned to every per-run record
// (one per (matrix, preconditioner/eps_prime) point).  The adaptive runs
// thread their run_id into the per-panel log so the two CSVs can be joined.
static long g_next_run_id = 0;

// ─── Context for the append-only per-panel log (INSTRUMENTATION §1) ──
struct PanelLogCtx {
    FILE*       fp            = nullptr;
    long        run_id        = 0;
    const char* matrix_id     = "";
    const char* spectrum_type = "";
    int         n             = 0;
    double      kappa         = 0.0;
    int         nb            = 0;
    double      eps_prime     = 0.0;
    double      c_underflow   = 0.0;
};

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
        h_row[i] = r - 1;
        h_col[i] = c - 1;
        h_val[i] = v;
    }
    fclose(f);

    int n = M;
    size_t dense_bytes = sizeof(double) * (size_t)n * n;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    if (dense_bytes * 3 > free_mem) {
        fprintf(stderr, "Insufficient GPU memory: need ~%.1f GB, have %.1f GB free\n",
                (double)(dense_bytes * 3) / (1024.0*1024.0*1024.0),
                (double)free_mem / (1024.0*1024.0*1024.0));
        free(h_row); free(h_col); free(h_val);
        return -1;
    }

    double* dA;
    CUDA_CHECK(cudaMalloc(&dA, dense_bytes));
    CUDA_CHECK(cudaMemset(dA, 0, dense_bytes));

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

    int pad_to = 256;
    int n_padded = ((n + pad_to - 1) / pad_to) * pad_to;

    if (n_padded != n) {
        printf("  Padding %d -> %d (multiple of %d)\n", n, n_padded, pad_to);

        size_t padded_bytes = sizeof(double) * (size_t)n_padded * n_padded;
        double* dA_padded;
        CUDA_CHECK(cudaMalloc(&dA_padded, padded_bytes));
        CUDA_CHECK(cudaMemset(dA_padded, 0, padded_bytes));

        CUDA_CHECK(cudaMemcpy2D(
            dA_padded,  sizeof(double) * n_padded,
            dA,         sizeof(double) * n,
            sizeof(double) * n, n,
            cudaMemcpyDeviceToDevice));

        int n_extra = n_padded - n;
        int bl2 = 256, gr2 = (n_extra + bl2 - 1) / bl2;
        set_diagonal_f64_kernel<<<gr2, bl2>>>(dA_padded, n_padded, n, n_padded, 1.0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(dA));
        dA = dA_padded;
        n  = n_padded;
    }

    *d_A = dA;
    *out_n = n;
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
    constexpr int NB = SPOTRF_PANEL_IB;

    extern __shared__ char smem_raw[];
    T* sA          = reinterpret_cast<T*>(smem_raw);
    T* dot_scratch = sA + NB * (NB + 1);

    const int tid     = threadIdx.x;
    const int lane    = tid & (warpSize - 1);
    const int warp_id = tid / warpSize;
    const int nwarps  = (blockDim.x + warpSize - 1) / warpSize;

    if (tid == 0) *info = 0;

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
    constexpr int NB = SPOTRF_PANEL_IB;
    int nwarps = 1;
    size_t smem = (NB * NB + nwarps + NB) * sizeof(T);
    potf2_panel_kernel<T><<<1, 32, smem, stream>>>(d_A, n, lda, d_info, beta, delta);
    CUDA_CHECK(cudaGetLastError());
}

inline void spotf2_panel(float* d_A, int n, int lda, float beta, float delta,
                         int* d_info, cudaStream_t stream = 0) {
    potf2_panel<float>(d_A, n, lda, beta, delta, d_info, stream);
}

// ═══════════════════════════════════════════════════════════════════
//  Diagonal perturbation kernels
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

inline void spotrf_panel(float* d_A, int n, int lda,
                         float beta, float delta,
                         cublasHandle_t blas_handle, int* d_info,
                         cudaStream_t stream = 0) {
    potrf_panel<float>(d_A, n, lda, beta, delta, blas_handle, d_info, stream);
}

// ═══════════════════════════════════════════════════════════════════
//  Cast / zero kernels
// ═══════════════════════════════════════════════════════════════════

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

__global__ void accumulate_bf16_into_f32_matrix_kernel(
    float* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int mr, int ld_dst, int ld_src,
    float alpha) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)mr * mr) return;
    int col = (int)(idx / mr);
    int row = (int)(idx % mr);
    dst[(size_t)col * ld_dst + row] += alpha * __bfloat162float(src[(size_t)col * ld_src + row]);
}

void accumulate_bf16_into_f32_matrix(float* dst, const __nv_bfloat16* src,
                                      int mr, int ld_dst, int ld_src,
                                      float alpha, cudaStream_t stream = 0) {
    size_t tot = (size_t)mr * mr;
    accumulate_bf16_into_f32_matrix_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, stream>>>(
        dst, src, mr, ld_dst, ld_src, alpha);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void accumulate_bf16_into_f64_matrix_kernel(
    double* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int mr, int ld_dst, int ld_src,
    double alpha) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)mr * mr) return;
    int col = (int)(idx / mr);
    int row = (int)(idx % mr);
    dst[(size_t)col * ld_dst + row] += alpha * (double)__bfloat162float(src[(size_t)col * ld_src + row]);
}

void accumulate_bf16_into_f64_matrix(double* dst, const __nv_bfloat16* src,
                                      int mr, int ld_dst, int ld_src,
                                      double alpha, cudaStream_t stream = 0) {
    size_t tot = (size_t)mr * mr;
    accumulate_bf16_into_f64_matrix_kernel<<<(unsigned)((tot + 255) / 256), 256, 0, stream>>>(
        dst, src, mr, ld_dst, ld_src, alpha);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void accumulate_f32_into_f64_kernel(
    double* __restrict__ dst,
    const float* __restrict__ src,
    size_t count,
    double alpha) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] += alpha * (double)src[i];
}

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
// ═══════════════════════════════════════════════════════════════════

template <typename T>
static __forceinline__ __device__ __host__ float get_epsilon() {
    if constexpr (std::is_same_v<double, T>)           return 1.11022302e-16f;
    if constexpr (std::is_same_v<float, T>)            return 1.19209290e-07f;
    if constexpr (std::is_same_v<__half, T>)           return 9.76562500e-04f;
    if constexpr (std::is_same_v<__nv_fp8_e4m3, T>)   return 0.125f;
    return 1.19209290e-07f;
}

template <typename T>
static __forceinline__ __device__ __host__ float get_smallest_subnormal() {
    if constexpr (std::is_same_v<double, T>)           return 0.0f;
    if constexpr (std::is_same_v<float, T>)            return 1.40129846e-45f;
    if constexpr (std::is_same_v<__half, T>)           return 5.96046448e-08f;
    if constexpr (std::is_same_v<__nv_fp8_e4m3, T>)   return 1.95312500e-03f;
    return 1.40129846e-45f;
}

static constexpr float TF32_EPSILON        = 9.76562500e-04f;
static constexpr float TF32_SUBNORMAL_MIN  = 1.40129846e-45f;

static constexpr double FP64_EPSILON       = 1.1102230246251565e-16;
static constexpr double FP64_SUBNORMAL_MIN = 4.9406564584124654e-324;

static __host__ float get_epsilon_for_prec(PrecisionChoice p) {
    switch (p) {
        case PREC_FP8:  return get_epsilon<__nv_fp8_e4m3>();
        case PREC_FP16: return get_epsilon<__half>();
        case PREC_TF32: return TF32_EPSILON;
        case PREC_FP32: return get_epsilon<float>();
        case PREC_FP64: return (float)FP64_EPSILON;
        default:        return get_epsilon<float>();
    }
}

static __host__ float get_smallest_subnormal_for_prec(PrecisionChoice p) {
    switch (p) {
        case PREC_FP8:  return get_smallest_subnormal<__nv_fp8_e4m3>();
        case PREC_FP16: return get_smallest_subnormal<__half>();
        case PREC_TF32: return TF32_SUBNORMAL_MIN;
        case PREC_FP32: return get_smallest_subnormal<float>();
        case PREC_FP64: return 0.0f;
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
//  Adaptive precision selection
// ═══════════════════════════════════════════════════════════════════
__global__ void update_diag_and_check_kernel(
    double* __restrict__ updated_diag,
    const double* __restrict__ orig_diag,
    double* __restrict__ spent,            // NEW: committed backward error per trailing row
    const double* __restrict__ L10,
    int trailing_start, int M, int K, int ldc,
    const float* __restrict__ g_coeff,     // NEW: [NFMT], g_P = 2*eps_P
    const float* __restrict__ omega,       // NEW: [NFMT], omega_P = c*eps_P^2*sub_P^2
    float eps_prime,                       // requested bound f_req
    float chol_reserve,                    // NEW: b*eps_h  (FP64 Cholesky reserve coeff)
    float prev_g_coeff,                    // NEW: 2*eps_{P_prev} for the panel just applied (0 on first step / if prev was FP64-exact)
    unsigned int* __restrict__ block_masks)
{
    extern __shared__ unsigned int smask[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_mask = 0x1Fu;

    if (gid < M) {
        int diag_idx = trailing_start + gid;

        double sum_sq = 0.0;
        for (int k = 0; k < K; k++) {
            double v = L10[(size_t)k * ldc + gid];
            sum_sq += v * v;
        }
        double new_diag = updated_diag[diag_idx] - sum_sq;
        updated_diag[diag_idx] = new_diag;

        // (3) accumulate committed error from the panel just applied (relative truncation)
        if (prev_g_coeff > 0.0f)
            spent[diag_idx] += (double)prev_g_coeff * sum_sq;

        double a_tt = orig_diag[diag_idx];
        // (2) budget net of Cholesky reserve and (3) already-committed error
        double rhs = ((double)eps_prime - (double)chol_reserve) * a_tt - spent[diag_idx];

        // (1) per-format: g_P*aᵘ + omega_P < rhs   (additive underflow, factor-2 in g_P)
        if (!((double)g_coeff[0]*new_diag + (double)omega[0] < rhs)) local_mask &= ~0x01u;
        if (!((double)g_coeff[1]*new_diag + (double)omega[1] < rhs)) local_mask &= ~0x02u;
        if (!((double)g_coeff[2]*new_diag + (double)omega[2] < rhs)) local_mask &= ~0x04u;
        if (!((double)g_coeff[3]*new_diag + (double)omega[3] < rhs)) local_mask &= ~0x08u;
        if (!((double)g_coeff[4]*new_diag + (double)omega[4] < rhs)) local_mask &= ~0x10u;
    }

    smask[tid] = local_mask;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) smask[tid] &= smask[tid+s]; __syncthreads(); }
    if (tid == 0) block_masks[blockIdx.x] = smask[0];
}

__global__ void reduce_and_kernel(
    const unsigned int* __restrict__ block_masks,
    int num_blocks,
    unsigned int* __restrict__ result) {

    extern __shared__ unsigned int smask[];
    int tid = threadIdx.x;

    unsigned int local_mask = 0x1Fu;
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
    double* d_spent,                 // NEW: length-n device array, zeroed once before the factorization
    const double* d_L10,
    int trailing_start,
    int M,
    int K,
    int ldc,
    float c,
    float eps_prime,
    float chol_reserve,              // NEW: b * eps_h
    PrecisionChoice prev_prec,       // NEW: precision of the panel GEMM just applied (PREC_FP64 => exact, no committed error)
    bool use_mx,
    float* d_g_coeff,                // NEW: device [NFMT], persistent scratch
    float* d_omega,                  // NEW: device [NFMT], persistent scratch
    unsigned int* d_block_masks,
    unsigned int* d_result_mask,
    cudaStream_t stream = 0)
{
    if (M <= 0) return PREC_FP64;

    // ---- build per-format coefficients on host, then upload ----
    // g_P = 2 * eps_P   (SYRK factor of 2; drop the 2.0f if get_epsilon_for_prec already bakes it in)
    // omega_P = c * eps_P^2 * sub_P^2   (additive underflow floor; negligible, but correct)
    const PrecisionChoice fmts[5] =
        { PREC_FP8, PREC_FP16, PREC_TF32, PREC_FP32, PREC_FP64 };

    float h_g[5], h_omega[5];
    for (int p = 0; p < 5; ++p) {
        float eps = get_epsilon_for_prec(fmts[p]);
        float sub = get_smallest_subnormal_for_prec(fmts[p]);
        h_g[p]     = 2.0f * eps;
        h_omega[p] = c * eps * eps * sub * sub;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_g_coeff, h_g, 5 * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_omega, h_omega, 5 * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    // committed-error coefficient for the panel just applied
    float prev_g = (prev_prec == PREC_FP64)
                       ? 0.0f                                 // FP64 GEMM: ~exact, nothing committed
                       : 2.0f * get_epsilon_for_prec(prev_prec);

    int block_size = 256;
    int num_blocks = (M + block_size - 1) / block_size;

    update_diag_and_check_kernel<<<num_blocks, block_size,
                                   block_size * sizeof(unsigned int), stream>>>(
        d_updated_diag, d_orig_diag, d_spent, d_L10,
        trailing_start, M, K, ldc,
        d_g_coeff, d_omega,
        eps_prime, chol_reserve, prev_g,
        d_block_masks);
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

    if (mask & 0x01u) return PREC_FP8;
    if (mask & 0x02u) return PREC_FP16;
    if (mask & 0x04u) return PREC_TF32;
    if (mask & 0x08u) return PREC_FP32;
    return PREC_FP64;
}

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
    return PREC_FP32;  // placeholder — adaptive-f32 path unused now
}

// ═══════════════════════════════════════════════════════════════════
//  Ill-conditioned SPD matrix generation
// ═══════════════════════════════════════════════════════════════════

void generate_spd(double* d_A, int n, int lda, double cond, eigenvalue_dist_t dist,
                  cublasHandle_t cb, cusolverDnHandle_t cs, cudaStream_t stream,
                  double shape_p = 2.0, int num_clusters = 4,
                  unsigned long long seed = 42ULL) {
    size_t nn = (size_t)n * n;
    double* d_B;
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(double) * nn));

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
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
                srand((unsigned)seed);
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
//  Quantization kernels — FP32 → low precision
// ═══════════════════════════════════════════════════════════════════

#define FP8_E4M3_MAX 448.0f

__global__ void fp32_to_fp8_tiled_kernel(
    __nv_fp8_e4m3* __restrict__ dst,
    const float*   __restrict__ src,
    float*         __restrict__ scales,
    int rows, int cols,
    int ld_src, int ld_dst)
{
    __shared__ float tile[QUANT_TILE][QUANT_TILE + 1];
    __shared__ float smem_max[QUANT_TILE];

    int row_base = blockIdx.x * QUANT_TILE;
    int col_base = blockIdx.y * QUANT_TILE;
    int tr = threadIdx.x;
    int tc = threadIdx.y;

    int src_row = row_base + tr;
    int src_col = col_base + tc;

    float v = 0.0f;
    if (src_row < rows && src_col < cols)
        v = src[(size_t)src_col * ld_src + src_row];

    tile[tc][tr] = v;
    __syncthreads();

    float local_abs = fabsf(tile[tr][tc]);

    #pragma unroll
    for (int offset = QUANT_TILE / 2; offset > 0; offset >>= 1)
        local_abs = fmaxf(local_abs, __shfl_xor_sync(0xffffffff, local_abs, offset));

    if (tc == 0)
        smem_max[tr] = local_abs;
    __syncthreads();

    float tile_max = 0.0f;
    if (tc == 0) {
        tile_max = smem_max[tr];
        #pragma unroll
        for (int offset = QUANT_TILE / 2; offset > 0; offset >>= 1)
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, offset));

        if (tr == 0) {
            float scale = (tile_max > 9.76562500e-04f*16.0f) ? tile_max / FP8_E4M3_MAX : 1.0f;
            int scale_cols = (cols + QUANT_TILE - 1) / QUANT_TILE;
            scales[blockIdx.x * scale_cols + blockIdx.y] = scale;
            smem_max[0] = scale;
        }
    }
    __syncthreads();

    float scale = smem_max[0];
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

    // Output coords are swapped: tile[tr][tc] is the value this thread
    // SHOULD write — it was loaded by the partner thread that read from
    // (row_base+tc, col_base+tr) in column-major source space.
    int dst_row = row_base + tc;
    int dst_col = col_base + tr;

    if (dst_row < rows && dst_col < cols) {
        float w = tile[tr][tc];
        dst[(size_t)dst_row * ld_dst + dst_col] =
            __float2half(fminf(fmaxf(w, -65504.f), 65504.f));
    }
}

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
//  hasNaN check on a square matrix
// ═══════════════════════════════════════════════════════════════════

__global__ void hasNaN_kernel(const double* __restrict__ data, int n, int lda, int* has_nan) {
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

// ═══════════════════════════════════════════════════════════════════
//  GemmWorkspace — per-call buffers for the adaptive Cholesky.
//  cuBLASLt handle/workspace lives in the global CublasLtContext.
// ═══════════════════════════════════════════════════════════════════

struct GemmWorkspace {
    // Row-major FP16 or FP8 staging panel.
    void*  fp_staging;
    size_t fp_staging_bytes;

    // Per-row scales for FP8 path (OUTER_VEC_32F mode).
    // One FP32 per row of A.  Sized for the largest panel: n rows max.
    float* scales;
    size_t scales_bytes;

    // FP32 temp for low-prec GEMM output when trailing matrix is FP64
    // (used by FP16, TF32, FP32 branches).
    float* gemm_temp;
    size_t gemm_temp_bytes;

    // BF16 temp for FP8 GEMM output (cuBLASLt FP8 with row-wise scaling
    // produces BF16/FP16 output, not FP32).
    __nv_bfloat16* gemm_temp_bf16;
    size_t          gemm_temp_bf16_bytes;

    // FP32 cast of the FP64 panel before low-prec quantization.
    float* panel_f32;
    size_t panel_f32_bytes;

    void allocate(int n, int nb, bool need_f64_temps) {
        fp_staging_bytes = sizeof(__half) * (size_t)n * nb;
        CUDA_CHECK(cudaMalloc(&fp_staging, fp_staging_bytes));

        // Row-wise FP8 scaling: one FP32 per row, up to n rows.
        scales_bytes  = sizeof(float) * (size_t)n;
        CUDA_CHECK(cudaMalloc(&scales, scales_bytes));

        // FP8 GEMM output: cuBLASLt's row-wise scaled FP8 produces
        // BF16/FP16, not FP32.  Allocate BF16 trailing-matrix-sized buffer.
        gemm_temp_bf16_bytes = sizeof(__nv_bfloat16) * (size_t)n * n;
        CUDA_CHECK(cudaMalloc(&gemm_temp_bf16, gemm_temp_bf16_bytes));

        gemm_temp = nullptr;     gemm_temp_bytes = 0;
        panel_f32 = nullptr;     panel_f32_bytes = 0;
        if (need_f64_temps) {
            gemm_temp_bytes = sizeof(float) * (size_t)n * n;
            CUDA_CHECK(cudaMalloc(&gemm_temp, gemm_temp_bytes));
            panel_f32_bytes = sizeof(float) * (size_t)n * nb;
            CUDA_CHECK(cudaMalloc(&panel_f32, panel_f32_bytes));
        }
    }

    void free() {
        if (fp_staging)     { cudaFree(fp_staging);     fp_staging     = nullptr; }
        if (scales)         { cudaFree(scales);         scales         = nullptr; }
        if (gemm_temp)      { cudaFree(gemm_temp);      gemm_temp      = nullptr; }
        if (gemm_temp_bf16) { cudaFree(gemm_temp_bf16); gemm_temp_bf16 = nullptr; }
        if (panel_f32)      { cudaFree(panel_f32);      panel_f32      = nullptr; }
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

// 2) Mixed FP16 Cholesky preconditioner (baseline)
//    Uses cuBLASLt FP16 GEMM with FP32 accumulation.
void chol_blocked_mixed_fp16(float* d_A, int n, int nb,
                              GemmWorkspace& ws,
                              cudaStream_t stream = 0)
{
    cusolverDnHandle_t sv; cublasHandle_t bl;
    CUSOLVER_CHECK(cusolverDnCreate(&sv)); CUBLAS_CHECK(cublasCreate(&bl));
    CUBLAS_CHECK(cublasSetStream(bl, stream));
    CUSOLVER_CHECK(cusolverDnSetStream(sv, stream));

    int lw = 0;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(
        sv, CUBLAS_FILL_MODE_LOWER, nb, d_A, n, &lw));
    float* dW; int* dI;
    CUDA_CHECK(cudaMalloc(&dW, sizeof(float) * std::max(lw, 1)));
    CUDA_CHECK(cudaMalloc(&dI, sizeof(int)));

    for (int j = 0; j < n; j += nb) {
        int jb = std::min(nb, n - j);
        float* Ajj = d_A + (size_t)j * n + j;

        CUSOLVER_CHECK(cusolverDnSpotrf(
            sv, CUBLAS_FILL_MODE_LOWER, jb, Ajj, n, dW, lw, dI));
        if (j + jb >= n) break;

        int   mr = n - j - jb;
        float one_f = 1.f;
        float* Ap = d_A + (size_t)j      * n + (j + jb);
        float* At = d_A + (size_t)(j+jb) * n + (j + jb);
        CUBLAS_CHECK(cublasStrsm(bl, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                 mr, jb, &one_f, Ajj, n, Ap, n));

        // FP32 panel (col-major, ld=n) → __half row-major (ld=jb)
        __half* p16 = reinterpret_cast<__half*>(ws.fp_staging);
        launch_convert<__half>(p16, Ap, ws.scales,
                                mr, jb, n, jb, stream);

        // FP16 GEMM via cuBLASLt: At -= p16 * p16^T
        g_fp16_gemm.run(mr, mr, jb,
                        p16, jb,
                        p16, jb,
                        At,  n,
                        -1.f, 1.f, stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dW)); CUDA_CHECK(cudaFree(dI));
    CUBLAS_CHECK(cublasDestroy(bl));
    CUSOLVER_CHECK(cusolverDnDestroy(sv));
}

// 3) Adaptive-precision Hybrid Cholesky
struct AdaptiveStats {
    int    step_counts[PREC_COUNT];
    double flops[PREC_COUNT];
    double total_flops;
    int    n_panels;          // number of panel steps with a Schur update
    int    n_nonpos_panels;   // panels where the pivot floor fired (nonpos_pivot)
};

float cuda_event_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// ═══════════════════════════════════════════════════════════════════
//  Adaptive-precision Hybrid Cholesky — templated on HighT
//
//  HighT = double:  A stored in FP64, panel factorized with potf2_panel<double>,
//                   low-prec GEMMs output to temp float buffer then
//                   accumulated into the double trailing matrix.
//                   FP64 GEMM uses cublasDgemm directly.
//
//  HighT = float:   Original behaviour — everything in FP32.
//
//  Low-precision GEMMs all go through cuBLASLt:
//    FP8  → g_fp8_gemm   (1-D block scaled, 128-along-K)
//    FP16 → g_fp16_gemm
//    TF32/FP32 → cublasSgemm
//    FP64 → cublasDgemm (only meaningful for HighT=double)
// ═══════════════════════════════════════════════════════════════════

template <typename HighT>
void chol_blocked_adaptive(HighT* d_A, int n, int nb,
                           float c, float eps_prime, bool use_mx,
                           bool verbose = false,
                           AdaptiveStats* out_stats = nullptr,
                           const PanelLogCtx* panel_log = nullptr) {
    static_assert(std::is_same_v<HighT, float> || std::is_same_v<HighT, double>,
                  "HighT must be float or double");
    constexpr bool is_f64 = std::is_same_v<HighT, double>;

    cublasHandle_t bl;
    CUBLAS_CHECK(cublasCreate(&bl));

    GemmWorkspace ws;
    ws.allocate(n, nb, /*need_f64_temps=*/is_f64);

    // Diagonal buffers — always double for precision selection
    double *d_orig_diag, *d_updated_diag, *d_spent;
    float *d_g_coeff, *d_omega;
    CUDA_CHECK(cudaMalloc(&d_g_coeff, sizeof(float) * 5));
    CUDA_CHECK(cudaMalloc(&d_omega,   sizeof(float) * 5));
    CUDA_CHECK(cudaMalloc(&d_orig_diag,    sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_updated_diag, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_spent, sizeof(double) * n));
    CUDA_CHECK(cudaMemset(d_spent, 0, sizeof(double) * n));  // zero once at start; updated incrementally on device

    int* dI_panel;
    CUDA_CHECK(cudaMalloc(&dI_panel, sizeof(int)));

#ifdef COUNT_UNDERFLOW
    // Per-row FP8 underflow tally for the whole L matrix, accumulated
    // across every panel step that quantizes that row (a trailing row
    // gets FP8-quantized once per step until it becomes part of a panel).
    int* d_row_underflow;
    CUDA_CHECK(cudaMalloc(&d_row_underflow, sizeof(int) * n));
    CUDA_CHECK(cudaMemset(d_row_underflow, 0, sizeof(int) * n));
#endif

    // Extract diagonal (always into double)
    if constexpr (is_f64) {
        extract_diagonal_f64(d_orig_diag, d_A, n, n);
    } else {
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
    int    n_panels_logged   = 0;   // panels with a Schur update (per-panel rows)
    int    n_nonpos_panels   = 0;   // panels whose factorization floored a pivot

    // Perturbation parameters for the panel factorization.
    // These are keyed to the precision used in the PREVIOUS step's
    // trailing-matrix update — that's the step that introduced
    // quantization error into the current panel.
    //
    // FP16 and above are precise enough that no perturbation is needed:
    // we use a huge beta and zero delta so the floor only fires on a
    // truly indefinite pivot (the existing safety net in potf2_panel).
    //
    // FP8 is where perturbation actually earns its keep — quantization
    // error can drive trailing-matrix diagonals slightly negative.
    auto prec_to_beta_delta = [](PrecisionChoice p) -> std::pair<double, double> {
        switch (p) {
            case PREC_FP8:
                // Per-step FP8 epsilon: ~1.95e-3 (E4M3), inflated 16x for headroom.
                return {16.0 * 1.953125e-3, 16.0 * 1.953125e-3};
            default:
                // FP16, TF32, FP32, FP64: no perturbation.
                // Huge beta -> row_term ~= 0; zero delta -> only floor on
                // ajj <= 0 (the safety net inside potf2_panel).
                return {99999999.0, 0.0};
        }
    };

    // `prev_prec` is updated at the END of each iteration to record what
    // precision was used for the trailing-matrix update of THIS step.
    // The NEXT iteration reads it to decide whether to perturb.
    PrecisionChoice prev_prec = PREC_FP64;   // initial: no perturbation

    for (int j = 0; j < n; j += nb) {
        int jb = std::min(nb, n - j);
        HighT* Ajj = d_A + (size_t)j * n + j;

        // Choose perturbation parameters based on the precision that
        // produced this panel's data.
        auto [beta_val, delta_val] = prec_to_beta_delta(prev_prec);

        // Panel factorization with the chosen perturbation.
        potrf_panel<HighT>(Ajj, jb, n,
                           (HighT)beta_val, (HighT)delta_val,
                           bl, dI_panel);

        // potf2_panel writes (last sub-block's) info: nonzero => a
        // non-positive pivot was floored to keep the panel SPD.
        int h_panel_info = 0;
        CUDA_CHECK(cudaMemcpy(&h_panel_info, dI_panel, sizeof(int),
                              cudaMemcpyDeviceToHost));
        const char* panel_status = (h_panel_info != 0) ? "nonpos_pivot" : "ok";

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

        // Precision selection
        int trailing_start = j + jb;
        PrecisionChoice prec;
        if constexpr (is_f64) {
            prec = select_precision_for_step_f64(
                d_updated_diag, d_orig_diag, d_spent, (double*)Ap,
                trailing_start, mr, jb, n,
                c, eps_prime, nb*exp2(-53.0), prev_prec, use_mx, d_g_coeff, d_omega,
                d_block_masks, d_result_mask);
        } else {
            prec = PREC_FP32;
        }

        double step_flops = 2.0 * (double)mr * (double)mr * (double)jb;
        prec_counts[prec]++;
        prec_flops[prec] += step_flops;
        n_panels_logged++;
        if (h_panel_info != 0) n_nonpos_panels++;

        // Per-panel log (INSTRUMENTATION §1) — one append-only row per panel
        // step, emitted right after precision is selected for the Schur update.
        // flops_executed is the GEMM count (~2x SYRK); flops_syrk_equiv = /2.
        if (panel_log && panel_log->fp) {
            fprintf(panel_log->fp,
                "%ld,%s,%s,%d,%.6e,%d,%.6e,%.6e,%d,%d,%.6e,%.6e,%s,%d,%s\n",
                panel_log->run_id, panel_log->matrix_id, panel_log->spectrum_type,
                panel_log->n, panel_log->kappa, panel_log->nb,
                panel_log->eps_prime, panel_log->c_underflow,
                j / nb,                          // panel_index (0-based)
                mr,                              // trailing_size m_j
                step_flops,                      // flops_executed f_j
                step_flops / 2.0,                // flops_syrk_equiv
                precision_name(prec),            // selected_precision
                mantissa_bits_for_prec(prec),    // mantissa_bits
                panel_status);                   // panel_status
            fflush(panel_log->fp);
        }

        if (verbose) {
            printf("  Step j=%d  mr=%d  jb=%d  -> %s  (%.3e FLOPs)\n",
                   j, mr, jb, precision_name(prec), step_flops);
        }

        prev_prec = prec;

        switch (prec) {
            case PREC_FP8: {
                float* panel32   = is_f64 ? ws.panel_f32 : (float*)Ap;
                int    ld_panel32 = is_f64 ? mr : n;

                if constexpr (is_f64) {
                    cast_f64_to_f32_matrix(ws.panel_f32, (double*)Ap, mr, jb, mr, n);
                }

                __nv_fp8_e4m3* p8 = reinterpret_cast<__nv_fp8_e4m3*>(ws.fp_staging);
                fp8bs1d::launch(p8, ws.scales, panel32,
                                mr, jb, ld_panel32, 0
#ifdef COUNT_UNDERFLOW
                                , d_row_underflow + trailing_start
#endif
                                );

                // GEMM into BF16 workspace.  C = 1 * L * L^T + 0
                g_fp8_gemm.run(mr, mr, jb,
                               p8, p8,
                               ws.scales, ws.scales,
                               ws.gemm_temp_bf16, /*ldd=*/mr,
                               1.f, 0.f, 0);

                // Accumulate -BF16 into the trailing matrix.
                if constexpr (is_f64) {
                    accumulate_bf16_into_f64_matrix(
                        (double*)At, ws.gemm_temp_bf16, mr, n, mr, -1.0);
                } else {
                    accumulate_bf16_into_f32_matrix(
                        (float*)At, ws.gemm_temp_bf16, mr, n, mr, -1.f);
                }
                break;
            }

            case PREC_FP16: {
                float* panel32    = is_f64 ? ws.panel_f32 : (float*)Ap;
                int    ld_panel32 = is_f64 ? mr : n;

                if constexpr (is_f64) {
                    cast_f64_to_f32_matrix(ws.panel_f32, (double*)Ap, mr, jb, mr, n);
                }

                __half* p16 = reinterpret_cast<__half*>(ws.fp_staging);
                launch_convert<__half>(p16, panel32, ws.scales,
                                        mr, jb, ld_panel32, jb);

                if constexpr (is_f64) {
                    CUDA_CHECK(cudaMemset(ws.gemm_temp, 0,
                                          sizeof(float) * (size_t)mr * mr));
                    g_fp16_gemm.run(mr, mr, jb,
                                    p16, jb,
                                    p16, jb,
                                    ws.gemm_temp, mr,
                                    1.f, 0.f, 0);
                    accumulate_f32_into_f64_matrix(
                        (double*)At, ws.gemm_temp, mr, n, mr, -1.0);
                } else {
                    g_fp16_gemm.run(mr, mr, jb,
                                    p16, jb,
                                    p16, jb,
                                    (float*)At, n,
                                    -1.f, 1.f, 0);
                }
                break;
            }

            case PREC_TF32: {
                if constexpr (is_f64) {
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
                    float neg = -1.f, one_f = 1.f;
                    CUBLAS_CHECK(cublasSetMathMode(bl, CUBLAS_PEDANTIC_MATH));
                    CUBLAS_CHECK(cublasSgemm(bl, CUBLAS_OP_N, CUBLAS_OP_T,
                        mr, mr, jb, &neg, (float*)Ap, n, (float*)Ap, n, &one_f, (float*)At, n));
                    CUBLAS_CHECK(cublasSetMathMode(bl, CUBLAS_DEFAULT_MATH));
                }
                break;
            }
        }
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
        out_stats->total_flops     = total_flops;
        out_stats->n_panels        = n_panels_logged;
        out_stats->n_nonpos_panels = n_nonpos_panels;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

#ifdef COUNT_UNDERFLOW
    {
        std::vector<int> h_row_underflow(n);
        CUDA_CHECK(cudaMemcpy(h_row_underflow.data(), d_row_underflow,
                              sizeof(int) * n, cudaMemcpyDeviceToHost));
        int worst_row = -1, worst_count = 0;
        long long total_underflow = 0;
        for (int i = 0; i < n; i++) {
            total_underflow += h_row_underflow[i];
            if (h_row_underflow[i] > worst_count) {
                worst_count = h_row_underflow[i];
                worst_row   = i;
            }
        }
        fprintf(stderr,
            "[fp8bs1d] whole-L underflow: worst row %d with %d underflowed elements, "
            "%lld total across all FP8 steps\n",
            worst_row, worst_count, total_underflow);
    }
    CUDA_CHECK(cudaFree(d_row_underflow));
#endif

    ws.free();
    CUDA_CHECK(cudaFree(dI_panel));
    CUDA_CHECK(cudaFree(d_orig_diag)); CUDA_CHECK(cudaFree(d_updated_diag));
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

// Device-side scalar helpers
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

    double *d_r, *d_z, *d_p, *d_Ap;
    CUDA_CHECK(cudaMalloc(&d_r,  sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_z,  sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_p,  sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_Ap, sizeof(double) * n));

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

    auto apply_Linv = [&](double* v) {
        if (d_L != nullptr) {
            CUBLAS_CHECK(cublasDtrsv(blas, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                     n, d_L, n, v, 1));
        }
    };

    auto apply_LTinv = [&](double* v) {
        if (d_L != nullptr) {
            CUBLAS_CHECK(cublasDtrsv(blas, CUBLAS_FILL_MODE_LOWER,
                                     CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                     n, d_L, n, v, 1));
        }
    };

    auto apply_precond = [&](const double* src, double* dst) {
        CUDA_CHECK(cudaMemcpy(dst, src, sizeof(double) * n, cudaMemcpyDeviceToDevice));
        apply_Linv(dst);
        apply_LTinv(dst);
    };

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

    double h_tol_sq_nbhat_sq = (tol * h_norm_bhat) * (tol * h_norm_bhat);

    CUBLAS_CHECK(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE));

    CUDA_CHECK(cudaMemcpy(d_r, d_b, sizeof(double) * n, cudaMemcpyDeviceToDevice));
    apply_precond(d_r, d_z);
    CUDA_CHECK(cudaMemcpy(d_p, d_z, sizeof(double) * n, cudaMemcpyDeviceToDevice));
    CUBLAS_CHECK(cublasDdot(blas, n, d_r, 1, d_z, 1, d_rz_old));

    int bl = 256, gr = (n + bl - 1) / bl;

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(t0));

    int iter = 0;
    int h_converged = 0;

    for (iter = 0; iter < max_iter; iter++) {
        CUBLAS_CHECK(cublasDgemv(blas, CUBLAS_OP_N, n, n,
                                 d_one, d_A, n, d_p, 1, d_zero, d_Ap, 1));

        CUBLAS_CHECK(cublasDdot(blas, n, d_p, 1, d_Ap, 1, d_pAp));

        compute_alpha_kernel<double><<<1, 1>>>(d_rz_old, d_pAp, d_alpha, d_neg_alpha);

        fused_xr_update_kernel<double><<<gr, bl>>>(d_x, d_r, d_p, d_Ap,
                                                    d_alpha, d_neg_alpha, n);

        apply_precond(d_r, d_z);

        CUBLAS_CHECK(cublasDdot(blas, n, d_r, 1, d_z, 1, d_rz_new));

        bool do_check = ((iter + 1) % check_interval == 0) ||
                        (iter + 1 == max_iter);
        if (do_check) {
            check_convergence_kernel<double><<<1, 1>>>(d_rz_new, h_tol_sq_nbhat_sq, d_converged);

            CUDA_CHECK(cudaMemcpyAsync(&h_converged, d_converged, sizeof(int),
                                        cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());

            if (h_converged) { iter++; break; }
        }

        compute_beta_and_swap_kernel<double><<<1, 1>>>(d_rz_old, d_rz_new, d_beta);

        fused_p_update_kernel<double><<<gr, bl>>>(d_p, d_z, d_beta, n);
    }

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

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
}

// ═══════════════════════════════════════════════════════════════════
//  Command-line parsing
// ═══════════════════════════════════════════════════════════════════
struct BenchmarkConfig {
    int n       = 4096;
    int nb      = 256;
    double cond = 1e6;
    int max_iter = 10000;
    double tol   = 1e-10;
    std::string csv_path;
    std::string panel_csv_path;   // per-panel log (INSTRUMENTATION §1); empty -> derived from csv_path
    float adapt_c         = 1.0f;
    float adapt_eps_prime = 0.1f;
    std::vector<float> eps_list;
    bool  adapt_use_mx    = false;
    bool  adapt_verbose   = false;
    double shape_p      = 2.0;
    int    num_clusters = 4;
    std::vector<std::string> mtx_paths;
    int    mtx_max_dim  = 40000;
    unsigned long long seed = 42ULL;
};

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --n <int>          Matrix size (default: 4096)\n");
    printf("  --nb <int>         Block size for Cholesky (default: 256)\n");
    printf("  --cond <float>     Condition number (default: 1e6)\n");
    printf("  --max_iter <int>   Max CG iterations (default: 10000)\n");
    printf("  --tol <float>      CG tolerance (default: 1e-10)\n");
    printf("  --csv <path>       Output per-run CSV file (optional)\n");
    printf("  --panel_csv <path> Output per-panel CSV file (default: <csv>_panel.csv)\n");
    printf("  --shape_p <float>  Exponent for poly_decay / power_law (default: 2.0)\n");
    printf("  --num_clusters <int>  Clusters for multi_cluster dist (default: 4)\n");
    printf("  --adapt_c <float>       Adaptive: constant c (default: 1.0)\n");
    printf("  --adapt_eps <float>     Adaptive: single eps_prime (default: 0.1)\n");
    printf("  --adapt_eps_list <csv>  Adaptive: comma-separated eps_prime sweep\n");
    printf("  --adapt_mx              [NO-OP, kept for CLI compatibility — FP8 always uses 1-D block scaling]\n");
    printf("  --adapt_verbose         Adaptive: print per-step precision choices\n");
    printf("  --mtx <path>       Matrix Market file (can specify multiple times)\n");
    printf("  --mtx_max_dim <n>  Max dimension for MTX files (default: 40000)\n");
    printf("  --seed <int>       RNG seed for synthetic SPD matrix generation (default: 42)\n");
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
        else if (strcmp(argv[i], "--panel_csv") == 0 && i + 1 < argc)
            cfg.panel_csv_path = argv[++i];
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
        else if (strcmp(argv[i], "--mtx") == 0 && i + 1 < argc)
            cfg.mtx_paths.push_back(argv[++i]);
        else if (strcmp(argv[i], "--mtx_max_dim") == 0 && i + 1 < argc)
            cfg.mtx_max_dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            cfg.seed = strtoull(argv[++i], nullptr, 10);
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
        data[idx] = (data[idx] * 2.0) - 1.0;
    }
}

// ─── Factorization backward error (INSTRUMENTATION §2 `achieved_berr`) ──
//  max_{i>=j} |a_ij - (L L^T)_ij| / sqrt(a_ii * a_jj).
//  One block per column j; threads stride over the lower-triangle rows.
//  Values are non-negative, so we max-reduce on the raw IEEE bit pattern
//  (monotone for non-negative doubles) with a single atomicMax.
__global__ void factor_berr_kernel(
    const double* __restrict__ A,     // original SPD matrix (full, col-major)
    const double* __restrict__ G,     // L L^T, lower triangle (col-major)
    const double* __restrict__ diagA, // diagonal of A, length n
    int n, int lda,
    unsigned long long* __restrict__ out_bits)
{
    int j = blockIdx.x;
    if (j >= n) return;
    double ajj = diagA[j];
    double local = 0.0;
    for (int i = j + threadIdx.x; i < n; i += blockDim.x) {
        double aij   = A[(size_t)j * lda + i];
        double gij   = G[(size_t)j * lda + i];
        double denom = sqrt(diagA[i] * ajj);
        double e     = (denom > 0.0) ? fabs(aij - gij) / denom : 0.0;
        local = fmax(local, e);
    }
    __shared__ double sm[256];
    sm[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] = fmax(sm[threadIdx.x], sm[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicMax(out_bits, (unsigned long long)__double_as_longlong(sm[0]));
}

static double compute_factor_berr(const double* d_A, const double* d_L,
                                  int n, cublasHandle_t blas)
{
    double *d_G = nullptr, *d_diag = nullptr;
    unsigned long long *d_bits = nullptr;
    CUDA_CHECK(cudaMalloc(&d_G,    sizeof(double) * (size_t)n * n));
    CUDA_CHECK(cudaMalloc(&d_diag, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_bits, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_bits, 0, sizeof(unsigned long long)));

    extract_diagonal_f64(d_diag, d_A, n, n);

    cublasPointerMode_t old_mode;
    cublasGetPointerMode(blas, &old_mode);
    cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST);
    double one = 1.0, zero = 0.0;
    // G = L L^T (lower triangle), L is the n x n lower-triangular factor.
    CUBLAS_CHECK(cublasDsyrk(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                             n, n, &one, d_L, n, &zero, d_G, n));
    cublasSetPointerMode(blas, old_mode);

    factor_berr_kernel<<<n, 256>>>(d_A, d_G, d_diag, n, n, d_bits);
    CUDA_CHECK(cudaGetLastError());

    unsigned long long h_bits = 0;
    CUDA_CHECK(cudaMemcpy(&h_bits, d_bits, sizeof(h_bits), cudaMemcpyDeviceToHost));
    cudaFree(d_G); cudaFree(d_diag); cudaFree(d_bits);

    double berr;
    memcpy(&berr, &h_bits, sizeof(double));   // reinterpret bits -> double
    return berr;
}

void run_benchmark_on_matrix(
    double* d_A64,
    int n,
    const char* matrix_name,
    const char* dist_label,
    BenchmarkConfig& cfg,
    cublasHandle_t cublas_handle,
    cusolverDnHandle_t cusolver_handle,
    cudaStream_t stream,
    FILE* csv_fp,
    FILE* per_panel_fp)
{
    size_t elems   = (size_t)n * n;
    size_t bytes64 = sizeof(double) * elems;
    size_t bytes32 = sizeof(float)  * elems;

    double *d_b64, *d_x64, *d_L64;
    float  *d_L32, *d_A32_copy;
    CUDA_CHECK(cudaMalloc(&d_b64, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_x64, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_L64, bytes64));
    CUDA_CHECK(cudaMalloc(&d_L32, bytes32));
    CUDA_CHECK(cudaMalloc(&d_A32_copy, bytes32));

    cast_f64_to_f32(d_A32_copy, d_A64, elems);
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        curandGenerator_t gen;
        CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
        CURAND_CHECK(curandGenerateUniformDouble(gen, d_b64, n));

        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        rescale_random_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_b64, n);

        CURAND_CHECK(curandDestroyGenerator(gen));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const char* baseline_names[] = { "None", "Dpotrf-FP64", "Mix-FP16" };

    printf("\n--- Baselines ---\n");
    printf("%-12s | %5s | %12s | %10s | %10s | %10s\n",
           "Precond", "Iters", "Rel Residual", "Setup (ms)", "PCG (ms)", "Total (ms)");
    for (int i = 0; i < 75; i++) printf("-");
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
                case 2: {
                    GemmWorkspace ws_tmp;
                    ws_tmp.allocate(n, cfg.nb, /*need_f64_temps=*/false);
                    chol_blocked_mixed_fp16(d_L32, n, cfg.nb, ws_tmp, stream);
                    ws_tmp.free();
                    break;
                }
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

        double backward_error = -1.0;
        {
            double norm_A = 0, norm_x = 0, norm_b = 0, norm_r = 0;
            double alpha = 1.0, beta = -1.0;

            CUBLAS_CHECK(cublasDnrm2(cublas_handle, n * n, d_A64, 1, &norm_A));
            CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_x64, 1, &norm_x));
            CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_b64, 1, &norm_b));

            double* d_res;
            CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_res, d_b64, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUBLAS_CHECK(cublasDgemv(cublas_handle, CUBLAS_OP_N, n, n,
                                     &alpha, d_A64, n,
                                     d_x64, 1,
                                     &beta, d_res, 1));
            CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_res, 1, &norm_r));
            backward_error = norm_r / (norm_A * norm_x + norm_b);

            printf("\n--- Stability Metrics ---\n");
            printf("  ||Ax - b||:       %.6e\n", norm_r);
            printf("  ||A||_F:           %.6e\n", norm_A);
            printf("  Backward Error:    %.6e\n", backward_error);

            CUDA_CHECK(cudaFree(d_res));
        }

        res.setup_ms = setup_ms;
        double total_ms = res.setup_ms + res.time_ms;

        // Convergence = PCG reached the relative-residual tolerance (criterion (b)
        // uses N_iter; here converged matches the solver's own stopping test, so a
        // non-converged run carries N_iter == max_iters).
        bool converged = (res.rel_residual <= cfg.tol);
        const char* status = converged ? "completed" : "cg_not_converged";

        printf("%-12s | %5d | %12.2e | %10.1f | %10.1f | %10.1f\n",
               baseline_names[p], res.iters, res.rel_residual,
               res.setup_ms, res.time_ms, total_ms);
        fflush(stdout);

        long run_id = g_next_run_id++;
        if (csv_fp) {
            // Columns after gemm_flops_total are the INSTRUMENTATION §2 fields.
            // Baselines carry no adaptive factor, so flops-by-precision and the
            // factorization achieved_berr are 0 / -1.  rhs is random (b in [-1,1]).
            fprintf(csv_fp,
                    "%d,%d,%.1e,0,%s,%s,%d,%.6e,%.3f,%.3f,%.3f,%s,%s,0,0,0,0,0,0,"
                    "%ld,%.6e,%.6e,%s,%.6e,%d,%s,%d,%d,%d,%d,%.6e,%.6e,%d\n",
                    n, cfg.nb, cfg.cond,
                    dist_label, baseline_names[p], res.iters, res.rel_residual,
                    res.setup_ms, res.time_ms, total_ms,
                    /*fact_time_source=*/"measured", /*cg_time_source=*/"measured",
                    run_id, /*c_underflow=*/0.0, /*kappa=*/cfg.cond,
                    /*rhs_spec=*/"random", /*solve_tol=*/cfg.tol,
                    /*max_iters=*/cfg.max_iter, status,
                    /*factorization_ok=*/1, /*break_panel=*/-1,
                    /*converged=*/(converged ? 1 : 0), /*N_iter=*/res.iters,
                    /*achieved_berr=*/-1.0, /*F_S=*/0.0,
                    /*simulated_construction=*/0);
            fflush(csv_fp);   // §0: per-run row as-it-happens (crash-safe)
        }
    }

    // Adaptive eps_prime sweep
    printf("\n--- Adaptive eps_prime sweep (%d values) ---\n",
           (int)cfg.eps_list.size());
    printf("%-12s | %5s | %12s | %30s | %10s | %10s | %10s | %8s | %8s | %8s | %8s | %8s\n",
           "eps_prime", "Iters", "Rel Residual", "estimated K(inv(L)*A*inv(L))",
           "Setup (ms)", "PCG (ms)", "Total (ms)",
           "FP8%", "FP16%", "TF32%", "FP32%", "FP64%");
    for (int i = 0; i < 145; i++) printf("-");
    printf("\n");

    for (int ei = 0; ei < (int)cfg.eps_list.size(); ei++) {
        float eps = cfg.eps_list[ei];

        CUDA_CHECK(cudaMemset(d_x64, 0, sizeof(double) * n));
        CUDA_CHECK(cudaMemcpy(d_L64, d_A64, bytes64, cudaMemcpyDeviceToDevice));

        AdaptiveStats adaptive_stats = {};
        long run_id = g_next_run_id++;

        PanelLogCtx panel_ctx;
        panel_ctx.fp            = per_panel_fp;
        panel_ctx.run_id        = run_id;
        panel_ctx.matrix_id     = matrix_name;
        panel_ctx.spectrum_type = dist_label;
        panel_ctx.n             = n;
        panel_ctx.kappa         = cfg.cond;
        panel_ctx.nb            = cfg.nb;
        panel_ctx.eps_prime     = (double)eps;
        panel_ctx.c_underflow   = (double)cfg.adapt_c;

        cudaEvent_t s0, s1;
        CUDA_CHECK(cudaEventCreate(&s0));
        CUDA_CHECK(cudaEventCreate(&s1));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(s0));

        chol_blocked_adaptive<double>(d_L64, n, cfg.nb,
                                      cfg.adapt_c, eps,
                                      cfg.adapt_use_mx, cfg.adapt_verbose,
                                      &adaptive_stats, &panel_ctx);

        zero_upper_f64(d_L64, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_has_nan = 0;
        {
            int* d_has_nan;
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

        // Factorization backward error (achieved_berr).  Only meaningful when
        // the factor is finite; the solve below is also skippable on failure,
        // but pcg_solve tolerates it and the run row is still recorded.
        double achieved_berr = h_has_nan ? -1.0
                              : compute_factor_berr(d_A64, d_L64, n, cublas_handle);

        CUDA_CHECK(cudaEventRecord(s1));
        CUDA_CHECK(cudaEventSynchronize(s1));
        float tmp_ms;
        CUDA_CHECK(cudaEventElapsedTime(&tmp_ms, s0, s1));
        double setup_ms = (double)tmp_ms;
        CUDA_CHECK(cudaEventDestroy(s0));
        CUDA_CHECK(cudaEventDestroy(s1));

#ifdef COND_APPROX
        double precond_cond = estimate_kappa(
            d_A64, d_L64, n, 100,
            cublas_handle, cusolver_handle);
#else
        double precond_cond = -1.0;
#endif

        CUDA_CHECK(cudaDeviceSynchronize());

        PCGResult res = pcg_solve(n, d_A64, d_b64, d_x64,
                                  d_L64, cfg.max_iter, cfg.tol);

        double backward_error = -1.0;
        {
            double norm_A = 0, norm_x = 0, norm_b = 0, norm_r = 0;
            double alpha = 1.0, beta = -1.0;

            CUBLAS_CHECK(cublasDnrm2(cublas_handle, n * n, d_A64, 1, &norm_A));
            CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_x64, 1, &norm_x));
            CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_b64, 1, &norm_b));

            double* d_res;
            CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_res, d_b64, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUBLAS_CHECK(cublasDgemv(cublas_handle, CUBLAS_OP_N, n, n,
                                     &alpha, d_A64, n,
                                     d_x64, 1,
                                     &beta, d_res, 1));
            CUBLAS_CHECK(cublasDnrm2(cublas_handle, n, d_res, 1, &norm_r));
            backward_error = norm_r / (norm_A * norm_x + norm_b);

            printf("\n--- Stability Metrics ---\n");
            printf("  ||Ax - b||:       %.6e\n", norm_r);
            printf("  ||A||_F:           %.6e\n", norm_A);
            printf("  Backward Error:    %.6e\n", backward_error);

            CUDA_CHECK(cudaFree(d_res));
        }
        res.setup_ms = setup_ms;
        double total_ms = res.setup_ms + res.time_ms;

        // Convergence = PCG reached the relative-residual tolerance (criterion (b)
        // uses N_iter; non-converged ⇒ N_iter == max_iters).  A completed
        // factorization that produced NaNs is still recorded with status "nan".
        bool converged = (res.rel_residual <= cfg.tol);
        bool factorization_ok = (h_has_nan == 0);
        const char* status = !factorization_ok ? "nan"
                           : (converged ? "completed" : "cg_not_converged");

        double fp8_pct  = adaptive_stats.total_flops > 0 ? 100.0 * adaptive_stats.flops[PREC_FP8]  / adaptive_stats.total_flops : 0;
        double fp16_pct = adaptive_stats.total_flops > 0 ? 100.0 * adaptive_stats.flops[PREC_FP16] / adaptive_stats.total_flops : 0;
        double tf32_pct = adaptive_stats.total_flops > 0 ? 100.0 * adaptive_stats.flops[PREC_TF32] / adaptive_stats.total_flops : 0;
        double fp32_pct = adaptive_stats.total_flops > 0 ? 100.0 * adaptive_stats.flops[PREC_FP32] / adaptive_stats.total_flops : 0;
        double fp64_pct = adaptive_stats.total_flops > 0 ? 100.0 * adaptive_stats.flops[PREC_FP64] / adaptive_stats.total_flops : 0;

        char eps_str[16];
        snprintf(eps_str, sizeof(eps_str), "%.6f", eps);

        printf("%-12s | %5d | %12.2e | %30.1f | %10.1f | %10.1f | %10.1f | %7.1f%% | %7.1f%% | %7.1f%% | %7.1f%% | %7.1f%%\n",
               eps_str, res.iters, res.rel_residual, precond_cond, res.setup_ms, res.time_ms,
               total_ms, fp8_pct, fp16_pct, tf32_pct, fp32_pct, fp64_pct);
        fflush(stdout);

        if (csv_fp) {
            fprintf(csv_fp,
                    "%d,%d,%.1e,%.6e,%s,%s,%d,%.6e,%.3f,%.3f,%.3f,%s,%s,"
                    "%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,"
                    "%ld,%.6e,%.6e,%s,%.6e,%d,%s,%d,%d,%d,%d,%.6e,%.6e,%d\n",
                    n, cfg.nb, cfg.cond, (double)eps,
                    dist_label, "Adaptive", res.iters, res.rel_residual,
                    res.setup_ms, res.time_ms, total_ms,
                    /*fact_time_source=*/"measured", /*cg_time_source=*/"measured",
                    adaptive_stats.flops[PREC_FP8],
                    adaptive_stats.flops[PREC_FP16],
                    adaptive_stats.flops[PREC_TF32],
                    adaptive_stats.flops[PREC_FP32],
                    adaptive_stats.flops[PREC_FP64],
                    adaptive_stats.total_flops,
                    run_id, (double)cfg.adapt_c, /*kappa=*/cfg.cond,
                    /*rhs_spec=*/"random", /*solve_tol=*/cfg.tol,
                    /*max_iters=*/cfg.max_iter, status,
                    /*factorization_ok=*/(factorization_ok ? 1 : 0),
                    /*break_panel=*/-1,
                    /*converged=*/(converged ? 1 : 0), /*N_iter=*/res.iters,
                    achieved_berr, /*F_S=*/adaptive_stats.flops[PREC_FP8],
                    /*simulated_construction=*/0);
            fflush(csv_fp);   // §0: per-run row as-it-happens (crash-safe)
        }
    }

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
    printf("  Adaptive params: c=%.4f  (FP8 always uses 1-D block scaling now)\n",
           cfg.adapt_c);
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

    // Initialize cuBLASLt context and persistent GEMM launchers
    g_lt_ctx.init(/*workspace_bytes=*/32 * 1024 * 1024);
    g_fp16_gemm.set_context(&g_lt_ctx);
    g_fp8_gemm .set_context(&g_lt_ctx);

    FILE* csv_fp = nullptr;
    if (!cfg.csv_path.empty()) {
        bool write_header = false;
        FILE* test = fopen(cfg.csv_path.c_str(), "r");
        if (!test) { write_header = true; }
        else { fseek(test, 0, SEEK_END); if (ftell(test) == 0) write_header = true; fclose(test); }
        csv_fp = fopen(cfg.csv_path.c_str(), "a");
        if (!csv_fp) {
            fprintf(stderr, "Cannot open CSV %s\n", cfg.csv_path.c_str());
        } else if (write_header) {
            fprintf(csv_fp, "n,nb,cond,eps_prime,dist,preconditioner,iterations,rel_residual,"
                    "setup_ms,pcg_ms,total_ms,fact_time_source,cg_time_source,"
                    "gemm_flops_fp8,gemm_flops_fp16,gemm_flops_tf32,gemm_flops_fp32,gemm_flops_fp64,gemm_flops_total,"
                    // INSTRUMENTATION §2 per-run fields:
                    "run_id,c_underflow,kappa,rhs_spec,solve_tol,max_iters,status,"
                    "factorization_ok,break_panel,converged,N_iter,achieved_berr,F_S,"
                    "simulated_construction\n");
        }
    }

    // Per-panel log (INSTRUMENTATION §1).  Default path: derive from csv_path
    // by inserting "_panel" before the extension.
    FILE* per_panel_fp = nullptr;
    {
        std::string panel_path = cfg.panel_csv_path;
        if (panel_path.empty() && !cfg.csv_path.empty()) {
            size_t dot = cfg.csv_path.rfind('.');
            panel_path = (dot == std::string::npos)
                       ? cfg.csv_path + "_panel.csv"
                       : cfg.csv_path.substr(0, dot) + "_panel" + cfg.csv_path.substr(dot);
        }
        if (!panel_path.empty()) {
            cfg.panel_csv_path = panel_path;
            bool write_header = false;
            FILE* test = fopen(panel_path.c_str(), "r");
            if (!test) { write_header = true; }
            else { fseek(test, 0, SEEK_END); if (ftell(test) == 0) write_header = true; fclose(test); }
            per_panel_fp = fopen(panel_path.c_str(), "a");
            if (!per_panel_fp) {
                fprintf(stderr, "Cannot open per-panel CSV %s\n", panel_path.c_str());
            } else if (write_header) {
                fprintf(per_panel_fp,
                        "run_id,matrix_id,spectrum_type,n,kappa,nb,eps_prime,c_underflow,"
                        "panel_index,trailing_size,flops_executed,flops_syrk_equiv,"
                        "selected_precision,mantissa_bits,panel_status\n");
            }
        }
    }

    // Branch 1: Matrix Market files
    if (!cfg.mtx_paths.empty()) {
        for (int mi = 0; mi < (int)cfg.mtx_paths.size(); mi++) {
            const char* mtx_path = cfg.mtx_paths[mi].c_str();

            std::string mtx_name = cfg.mtx_paths[mi];
            size_t slash = mtx_name.rfind('/');
            if (slash != std::string::npos) mtx_name = mtx_name.substr(slash + 1);
            size_t dot = mtx_name.rfind('.');
            if (dot != std::string::npos) mtx_name = mtx_name.substr(0, dot);

            printf("================================================================\n");
            printf("  Matrix Market: %s\n", mtx_name.c_str());
            printf("================================================================\n");

            double* d_A64 = nullptr;
            int n = 0;
            int rc = load_mtx_to_dense_f64(mtx_path, &d_A64, &n, cfg.mtx_max_dim);
            if (rc != 0) {
                printf("  SKIPPING (load failed)\n\n");
                continue;
            }

            int saved_n = cfg.n;
            cfg.n = n;

            char dist_label[256];
            snprintf(dist_label, sizeof(dist_label), "mtx:%s", mtx_name.c_str());

            double saved_cond = cfg.cond;
            cfg.cond = 0.0;

            run_benchmark_on_matrix(d_A64, n, mtx_name.c_str(), dist_label,
                                    cfg, cublas_handle, cusolver_handle,
                                    stream, csv_fp, per_panel_fp);

            cfg.n = saved_n;
            cfg.cond = saved_cond;

            CUDA_CHECK(cudaFree(d_A64));
            printf("\n");
        }
    }

    // Branch 2: Synthetic matrices
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

            printf("================================================================\n");
            printf("  Distribution: %s\n", dist_label);
            printf("================================================================\n");

            printf("Generating %d x %d SPD matrix (cond=%.1e, dist=%s) ... ",
                   n, n, cfg.cond, dist_label);
            fflush(stdout);
            generate_spd(d_A64, n, n, cfg.cond, dist, cublas_handle,
                         cusolver_handle, stream, cfg.shape_p, cfg.num_clusters,
                         cfg.seed);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            printf("done.\n");

            run_benchmark_on_matrix(d_A64, n, dist_label, dname,
                                    cfg, cublas_handle, cusolver_handle,
                                    stream, csv_fp, per_panel_fp);
            printf("\n");
        }

        CUDA_CHECK(cudaFree(d_A64));
    }

    if (csv_fp) {
        fclose(csv_fp);
        printf("Per-run results written to: %s\n", cfg.csv_path.c_str());
    }
    if (per_panel_fp) {
        fclose(per_panel_fp);
        printf("Per-panel results written to: %s\n", cfg.panel_csv_path.c_str());
    }

    printf("PCG tolerance: %.0e, max iterations: %d\n", cfg.tol, cfg.max_iter);

    // Cleanup cuBLASLt
    g_fp16_gemm.clear_cache();
    g_fp8_gemm .clear_cache();
    g_lt_ctx.destroy();

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));

    return 0;
}