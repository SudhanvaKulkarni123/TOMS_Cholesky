/**
 * cublas_ozaki_cholesky.cu
 *
 * Blocked RIGHT-LOOKING Cholesky with configurable pivoting, cuBLAS Ozaki
 * FP64 emulation, and optional power-of-2 diagonal equilibration.
 *
 * Right-looking structure:  POTRF -> TRSM -> GEMM  (per block-step)
 *
 * Three GEMM strategies compared:
 *   1) cuBLAS FP64 emulation with FIXED mantissa control (7 slices = 55 bits)
 *   2) cuBLAS FP64 emulation with DYNAMIC mantissa control (ADP)
 *   3) Custom adaptive: user kernel picks mantissa bits per block-step,
 *      then cuBLAS fixed-mantissa emulation executes the GEMM
 *
 * All three are compared against a native FP64 cuBLAS baseline.
 *
 * Build (CUDA 13.0u2+ required):
 *   nvcc -O3 -std=c++17 -arch=sm_90a emulated_chol.cu mmio.c -lcublas -lcusolver -lcurand -lcudart -o emulated_chol
 *
 * Run:
 *   ./emulated_chol [n] [nb] [cond] [dist] [pivot] [equil] [shape_p] [clusters] [csv] [mtx_path] [mtx_max_dim]
 *
 *   dist  1-9:  1=geometric 2=arithmetic 3={1,1/c,...}
 *               4={1,...,1,1/c} 5=random 6=half-half
 *               7=poly-decay 8=multi-cluster 9=power-law
 *   pivot 0-2:  0=static 1=dynamic-topk 2=dynamic-random
 *   equil 0-1:  0=compare without equil only
 *               1=compare both with and without equil
 *   shape_p:    exponent for poly-decay (dist=7) and power-law (dist=9), default 2.0
 *   clusters:   number of clusters for multi-cluster (dist=8), default 4
 *   csv:        path to CSV file for recording results (appends if exists)
 *   mtx_path:   optional Matrix Market file. If given, n / cond / dist / shape_p / clusters
 *               are ignored and the matrix is loaded from disk. The matrix must be
 *               real, symmetric, square, and SPD (verified via Cholesky probe).
 *               Maximum dimension is mtx_max_dim (default 40000) so it fits on the GPU.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>

extern "C" {
  #include "mmio.h"
}

// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
#define CUBLAS_CHECK(call) do { cublasStatus_t s=(call); if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} } while(0)
#define CUSOLVER_CHECK(call) do { cusolverStatus_t s=(call); if(s!=CUSOLVER_STATUS_SUCCESS){fprintf(stderr,"cuSOLVER %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} } while(0)
#define CURAND_CHECK(call) do { curandStatus_t s=(call); if(s!=CURAND_STATUS_SUCCESS){fprintf(stderr,"cuRAND %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);} } while(0)

// ---------------------------------------------------------------------------
// Helpers: mantissa bits <-> slice count
// sliceCount = ceildiv(mantissaBitCount + 1, 8)
// For 7 slices: mantissaBitCount = 7*8 - 1 = 55
// ---------------------------------------------------------------------------
static inline int bits_to_slices(int bits) { return (bits + 1 + 7) / 8; }
static inline int slices_to_bits(int slices) { return slices * 8 - 1; }

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------
enum eigenvalue_dist_t { DIST_GEOMETRIC=1, DIST_ARITHMETIC=2, DIST_ONE_FLAT=3,
                         DIST_FLAT_ONE=4, DIST_RANDOM=5, DIST_HALF_HALF=6,
                         DIST_POLY_DECAY=7, DIST_MULTI_CLUSTER=8, DIST_POWER_LAW=9 };
const char* dist_name(eigenvalue_dist_t d){
  switch(d){ case DIST_GEOMETRIC:return"geometric"; case DIST_ARITHMETIC:return"arithmetic";
    case DIST_ONE_FLAT:return"{1,1/c,...}"; case DIST_FLAT_ONE:return"{1,...,1/c}";
    case DIST_RANDOM:return"random"; case DIST_HALF_HALF:return"half-half";
    case DIST_POLY_DECAY:return"poly-decay"; case DIST_MULTI_CLUSTER:return"multi-cluster";
    case DIST_POWER_LAW:return"power-law"; default:return"?";}
}
enum pivot_mode_t { PIVOT_STATIC=0, PIVOT_DYNAMIC_TOPK=1, PIVOT_DYNAMIC_RAND=2 };
const char* pivot_name(pivot_mode_t p){
  switch(p){ case PIVOT_STATIC:return"static"; case PIVOT_DYNAMIC_TOPK:return"dyn-topk";
    case PIVOT_DYNAMIC_RAND:return"dyn-rand"; default:return"?";}
}

void print_Matrix(const double* A, int n, int lda) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%8.4f ", A[j * lda + i]);
    }
    printf("\n");
  }
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------
__global__ void zero_upper_kernel(double*A,int n,int lda){
  int r=blockIdx.x*blockDim.x+threadIdx.x, c=blockIdx.y*blockDim.y+threadIdx.y;
  if(r<n&&c<n&&c>r) A[(std::size_t)c*lda+r]=0.0;
}
void zero_upper(double*d,int n,int lda){
  dim3 b(32,32),g((n+31)/32,(n+31)/32);
  zero_upper_kernel<<<g,b>>>(d,n,lda); CUDA_CHECK(cudaGetLastError());
}

__global__ void extract_diag_kernel(const double*A,double*d,int n,int lda){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) d[i]=A[(std::size_t)i*lda+i];
}
void extract_diag(const double*A,double*d,int n,int lda,cudaStream_t s){
  extract_diag_kernel<<<(n+255)/256,256,0,s>>>(A,d,n,lda); CUDA_CHECK(cudaGetLastError());
}

__global__ void scale_col_kernel(double*A,int n,int lda,int col,double v){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) A[(std::size_t)col*lda+i]*=v;
}

__global__ void scale_row_kernel(double*A,int n,int lda,int row,double v){
  int j=blockIdx.x*blockDim.x+threadIdx.x;
  if(j<n) A[(std::size_t)j*lda+row]*=v;
}

__global__ void sym_perm_kernel(const double*A,double*B,const int*p,int n,int lda){
  int r=blockIdx.x*blockDim.x+threadIdx.x, c=blockIdx.y*blockDim.y+threadIdx.y;
  if(r<n&&c<n) B[(std::size_t)c*lda+r]=A[(std::size_t)p[c]*lda+p[r]];
}
void sym_perm_gpu(const double*A,double*B,const int*p,int n,int lda){
  dim3 b(32,32),g((n+31)/32,(n+31)/32);
  sym_perm_kernel<<<g,b>>>(A,B,p,n,lda); CUDA_CHECK(cudaGetLastError());
}

__global__ void swap_rows_kernel(double*A,int n,int lda,int ri,int rj){
  int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c<n){ auto a=(std::size_t)c*lda; double t=A[a+ri]; A[a+ri]=A[a+rj]; A[a+rj]=t; }
}
__global__ void swap_cols_kernel(double*A,int n,int lda,int ci,int cj){
  int r=blockIdx.x*blockDim.x+threadIdx.x;
  if(r<n){ double t=A[(std::size_t)ci*lda+r]; A[(std::size_t)ci*lda+r]=A[(std::size_t)cj*lda+r]; A[(std::size_t)cj*lda+r]=t; }
}
void sym_swap(double*d,int n,int lda,int i,int j,cudaStream_t s){
  if(i==j)return; int b=256,g=(n+255)/256;
  swap_rows_kernel<<<g,b,0,s>>>(d,n,lda,i,j);
  swap_cols_kernel<<<g,b,0,s>>>(d,n,lda,i,j);
  CUDA_CHECK(cudaGetLastError());
}
__global__ void swap_elem_kernel(double*a,int i,int j){
  if(threadIdx.x==0){double t=a[i];a[i]=a[j];a[j]=t;}
}
void swap_diag(double*d,int i,int j,cudaStream_t s){
  if(i==j)return; swap_elem_kernel<<<1,1,0,s>>>(d,i,j); CUDA_CHECK(cudaGetLastError());
}

// Equilibration kernel:  A[i,j] *= d_scale[i] * d_scale[j]
__global__ void equilibrate_kernel(double *A, const double *d_scale,
                                    const int n, const int lda) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < n && c < n)
    A[(std::size_t)c * lda + r] *= d_scale[r] * d_scale[c];
}

// Scale rows of L:  L[i, :] *= scale[i]   (undo equilibration)
__global__ void scale_rows_kernel(double *L, const double *scale,
                                   const int n, const int lda) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < n && c < n)
    L[(std::size_t)c * lda + r] *= scale[r];
}

// ---------------------------------------------------------------------------
// MTX loader support kernels
// ---------------------------------------------------------------------------
//
// Scatter COO triplets into a column-major dense matrix.  When the matrix is
// stored symmetrically (only the lower triangle in the file), each off-diagonal
// entry must be written twice: A[r,c] and A[c,r].
__global__ void coo_to_dense_f64_kernel(double* A,
                                         const int* row, const int* col,
                                         const double* val, int nnz,
                                         int M, int /*N*/, bool symmetric) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nnz) return;
  int r = row[idx];
  int c = col[idx];
  double v = val[idx];
  // Column-major: element (r, c) lives at A[c * lda + r]; here lda == M.
  A[(std::size_t)c * M + r] = v;
  if (symmetric && r != c) {
    A[(std::size_t)r * M + c] = v;
  }
}

// Set A[i, i] = val for i in [start, n_padded).  Used to extend a smaller
// SPD matrix with an identity block on the padded diagonal so the padded
// matrix stays SPD.
__global__ void set_diagonal_f64_kernel(double* A, int n_padded,
                                         int start, int lda, double val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int diag_idx = start + i;
  if (diag_idx >= n_padded) return;
  A[(std::size_t)diag_idx * lda + diag_idx] = val;
}

// ---------------------------------------------------------------------------
// Equilibration
// ---------------------------------------------------------------------------
void equilibrate_gpu(double *d_A, const int n, const int lda,
                     double *d_scale, cudaStream_t stream) {
  std::vector<double> h_diag(n);
  double *d_diag_tmp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_diag_tmp, sizeof(double) * n));
  extract_diag(d_A, d_diag_tmp, n, lda, stream);
  CUDA_CHECK(cudaMemcpyAsync(h_diag.data(), d_diag_tmp, sizeof(double) * n,
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(d_diag_tmp));

  std::vector<double> h_scale(n);
  for (int i = 0; i < n; i++) {
    if (h_diag[i] > 0.0) {
      double half_log2 = 0.5 * std::log2(h_diag[i]);
      int exponent = (int)std::round(half_log2);
      h_scale[i] = std::ldexp(1.0, -exponent);
    } else {
      h_scale[i] = 1.0;
    }
  }

  CUDA_CHECK(cudaMemcpyAsync(d_scale, h_scale.data(), sizeof(double) * n,
                              cudaMemcpyHostToDevice, stream));

  dim3 block(32, 32), grid((n + 31) / 32, (n + 31) / 32);
  equilibrate_kernel<<<grid, block, 0, stream>>>(d_A, d_scale, n, lda);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

__global__ void invert_scale_kernel(const double *scale, double *inv_scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) inv_scale[i] = 1.0 / scale[i];
}

void unequilibrate_L(double *d_L, const double *d_scale, const int n,
                     const int lda, cudaStream_t stream) {
  double *d_inv = nullptr;
  CUDA_CHECK(cudaMalloc(&d_inv, sizeof(double) * n));
  invert_scale_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_scale, d_inv, n);
  CUDA_CHECK(cudaGetLastError());

  dim3 block(32, 32), grid((n + 31) / 32, (n + 31) / 32);
  scale_rows_kernel<<<grid, block, 0, stream>>>(d_L, d_inv, n, lda);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(d_inv));
}

__global__ void permute_vec_kernel(const double *in, double *out,
                                    const int *perm, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[perm[i]];
}

// ---------------------------------------------------------------------------
// Panel pivoting
// ---------------------------------------------------------------------------
void panel_pivot(double*d_A,int n,int lda,double*d_do,double*d_du,
                 std::vector<int>&gp,int j,int jb,pivot_mode_t pm,cudaStream_t s){
  int trail=n-j; if(trail<=jb)return;
  std::vector<double> hd(trail);
  CUDA_CHECK(cudaMemcpyAsync(hd.data(),d_du+j,sizeof(double)*trail,cudaMemcpyDeviceToHost,s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  std::vector<int> li(trail); std::iota(li.begin(),li.end(),0);
  std::vector<int> sel(jb);
  if(pm==PIVOT_DYNAMIC_TOPK){
    std::partial_sort(li.begin(),li.begin()+jb,li.end(),[&](int a,int b){return hd[a]>hd[b];});
    for(int t=0;t<jb;t++) sel[t]=li[t];
  } else {
    std::mt19937 rng(42+j); std::vector<bool> taken(trail,false);
    for(int t=0;t<jb;t++){
      double tw=0; for(int k=0;k<trail;k++) if(!taken[k]) tw+=std::max(hd[k],0.0);
      if(tw<=0){for(int k=0;k<trail;k++) if(!taken[k]){sel[t]=k;taken[k]=true;break;}continue;}
      std::uniform_real_distribution<double> ud(0,tw); double r=ud(rng),cu=0; int pk=-1;
      for(int k=0;k<trail;k++){if(taken[k])continue;cu+=std::max(hd[k],0.0);if(cu>=r){pk=k;break;}}
      if(pk<0)pk=trail-1; sel[t]=pk; taken[pk]=true;
    }
  }
  std::sort(sel.begin(),sel.end());
  for(int t=0;t<jb;t++){
    int sl=sel[t]; if(sl==t)continue;
    int gi=j+t,gj=j+sl;
    sym_swap(d_A,n,lda,gi,gj,s);
    swap_diag(d_do,gi,gj,s); swap_diag(d_du,gi,gj,s);
    std::swap(gp[gi],gp[gj]);
    for(int u=t+1;u<jb;u++) if(sel[u]==t) sel[u]=sl;
  }
  CUDA_CHECK(cudaStreamSynchronize(s));
}

// ---------------------------------------------------------------------------
// MatrixMarket loader (real symmetric coordinate -> dense column-major).
//
// Returns 0 on success.  Returns -1 on any failure (file error, wrong type,
// matrix not square, dim above max_dim, GPU OOM, etc.).
//
// On success, *d_A is a freshly-allocated n*n column-major dense buffer
// holding the full (both triangles) matrix, and *out_n holds n.
//
// NOTE: this function only LOADS the matrix.  SPD verification and padding
// are done separately so that callers can fail fast on non-SPD inputs without
// paying for the padding allocation.
// ---------------------------------------------------------------------------
int load_mtx_to_dense_f64(const char* path, double** d_A, int* out_n,
                           int max_dim = 40000)
{
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "  Cannot open %s\n", path);
        return -1;
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        fprintf(stderr, "  Could not read Matrix Market banner in %s\n", path);
        fclose(f);
        return -1;
    }

    if (!mm_is_matrix(matcode) || !mm_is_coordinate(matcode) || !mm_is_real(matcode)) {
        fprintf(stderr, "  Only real coordinate matrices supported (got %s)\n", path);
        fclose(f);
        return -1;
    }

    bool symmetric = mm_is_symmetric(matcode);
    if (!symmetric) {
        fprintf(stderr, "  Matrix at %s is not symmetric (required for SPD)\n", path);
        fclose(f);
        return -1;

    }

    int M, N, nnz_file;
    if (mm_read_mtx_crd_size(f, &M, &N, &nnz_file) != 0) {
        fprintf(stderr, "  Could not read matrix size from %s\n", path);
        fclose(f);
        return -1;
    }

    if (M != N) {
        fprintf(stderr, "  Matrix is not square: %d x %d\n", M, N);
        fclose(f);
        return -1;
    }

    if (M > max_dim) {
        fprintf(stderr, "  Matrix too large: n=%d > %d max\n", M, max_dim);
        fclose(f);
        return -1;
    }

    printf("  Loading %s: n=%d, %d stored entries, symmetric\n",
           path, M, nnz_file);

    // Read COO triplets on host
    int*    h_row = (int*)malloc(sizeof(int)    * nnz_file);
    int*    h_col = (int*)malloc(sizeof(int)    * nnz_file);
    double* h_val = (double*)malloc(sizeof(double) * nnz_file);

    for (int i = 0; i < nnz_file; i++) {
        int r, c;
        double v;
        if (fscanf(f, "%d %d %lg", &r, &c, &v) != 3) {
            fprintf(stderr, "  Parse error at entry %d\n", i);
            free(h_row); free(h_col); free(h_val);
            fclose(f);
            return -1;
        }
        h_row[i] = r - 1;  // 1-based -> 0-based
        h_col[i] = c - 1;
        h_val[i] = v;
    }
    fclose(f);

    int n = M;
    std::size_t dense_bytes = sizeof(double) * (std::size_t)n * n;

    // GPU memory probe: we need at least the dense matrix plus a comparable
    // amount of headroom for the SPD probe and benchmark workspace.
    std::size_t free_mem = 0, total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    if (dense_bytes * 3 > free_mem) {
        fprintf(stderr, "  Insufficient GPU memory: need ~%.1f GB, have %.1f GB free\n",
                (double)(dense_bytes * 3) / (1024.0*1024.0*1024.0),
                (double)free_mem / (1024.0*1024.0*1024.0));
        free(h_row); free(h_col); free(h_val);
        return -1;
    }

    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, dense_bytes));
    CUDA_CHECK(cudaMemset(dA, 0, dense_bytes));

    int*    d_row = nullptr;
    int*    d_col = nullptr;
    double* d_val = nullptr;
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

    printf("  Dense matrix scattered: %d x %d  (%.2f GB)\n",
           n, n, (double)dense_bytes / (1024.0*1024.0*1024.0));
    return 0;
}

// ---------------------------------------------------------------------------
// SPD probe: factor a copy with cuSOLVER's native FP64 Cholesky.  If POTRF
// returns info > 0 the matrix has a non-positive-definite leading minor and
// is rejected.  Note this is a full factorization (~n^3/3 FLOPs); for n=40000
// expect tens of seconds.
// ---------------------------------------------------------------------------
bool check_spd_gpu(const double* d_A, int n, int lda,
                   cusolverDnHandle_t cs, cudaStream_t s) {
    std::size_t mb = sizeof(double) * (std::size_t)lda * n;
    double* d_test = nullptr;
    CUDA_CHECK(cudaMalloc(&d_test, mb));
    CUDA_CHECK(cudaMemcpyAsync(d_test, d_A, mb, cudaMemcpyDeviceToDevice, s));

    int ws = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cs, CUBLAS_FILL_MODE_LOWER,
                                                n, d_test, lda, &ws));
    double* d_ws = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ws, sizeof(double) * (std::size_t)ws));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    printf("  Running SPD probe (cuSOLVER POTRF, this may take a while for large n) ... ");
    fflush(stdout);

    CUSOLVER_CHECK(cusolverDnDpotrf(cs, CUBLAS_FILL_MODE_LOWER,
                                     n, d_test, lda, d_ws, ws, d_info));
    int info_h = 0;
    CUDA_CHECK(cudaMemcpyAsync(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    CUDA_CHECK(cudaFree(d_test));
    CUDA_CHECK(cudaFree(d_ws));
    CUDA_CHECK(cudaFree(d_info));

    if (info_h == 0) {
        printf("OK (matrix is SPD).\n");
        return true;
    } else {
        printf("FAILED (POTRF info=%d, leading minor not PD).\n", info_h);
        return false;
    }
}

// ---------------------------------------------------------------------------
// Pad an SPD matrix in place to the next multiple of pad_to, extending the
// padded diagonal with 1's so the result remains SPD ([A 0; 0 I] form).
// On success, *d_A may point to a freshly-allocated buffer and *n is the
// new (padded) dimension.  No-op if n is already a multiple of pad_to.
// ---------------------------------------------------------------------------
int pad_spd_to_multiple(double** d_A, int* n, int pad_to) {
    int n_orig = *n;
    int n_padded = ((n_orig + pad_to - 1) / pad_to) * pad_to;
    if (n_padded == n_orig) return 0;

    printf("  Padding %d -> %d (multiple of %d)\n", n_orig, n_padded, pad_to);

    std::size_t padded_bytes = sizeof(double) * (std::size_t)n_padded * n_padded;
    double* dA_padded = nullptr;
    CUDA_CHECK(cudaMalloc(&dA_padded, padded_bytes));
    CUDA_CHECK(cudaMemset(dA_padded, 0, padded_bytes));

    // Copy n_orig x n_orig block into top-left of padded matrix (col-major)
    CUDA_CHECK(cudaMemcpy2D(
        dA_padded, sizeof(double) * (std::size_t)n_padded,
        *d_A,      sizeof(double) * (std::size_t)n_orig,
        sizeof(double) * (std::size_t)n_orig, n_orig,
        cudaMemcpyDeviceToDevice));

    // Padded rows/cols get identity diagonal -> stays SPD
    int n_extra = n_padded - n_orig;
    int bl = 256, gr = (n_extra + bl - 1) / bl;
    set_diagonal_f64_kernel<<<gr, bl>>>(dA_padded, n_padded,
                                         n_orig, n_padded, 1.0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(*d_A));
    *d_A = dA_padded;
    *n   = n_padded;
    return 0;
}

// ---------------------------------------------------------------------------
// Helper: extract the stem (basename without extension) from a path, used
// for the CSV "dist" label when running on a Matrix Market file.
// ---------------------------------------------------------------------------
std::string filename_stem(const std::string& path) {
    std::string base = path;
    auto slash = base.find_last_of("/\\");
    if (slash != std::string::npos) base = base.substr(slash + 1);
    auto dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    return base;
}

// ---------------------------------------------------------------------------
// Generate SPD matrix
// ---------------------------------------------------------------------------
void generate_spd(double*d_A,int n,int lda,double cond,eigenvalue_dist_t dist,
                  cublasHandle_t cb,cusolverDnHandle_t cs,cudaStream_t stream,
                  double shape_p=2.0,int num_clusters=4){
  std::size_t nn=(std::size_t)n*n;
  double*d_B; CUDA_CHECK(cudaMalloc(&d_B,sizeof(double)*nn));
  curandGenerator_t gen;
  CURAND_CHECK(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen,42ULL));
  CURAND_CHECK(curandSetStream(gen,stream));
  CURAND_CHECK(curandGenerateNormalDouble(gen,d_B,nn,0.0,1.0));
  CURAND_CHECK(curandDestroyGenerator(gen));
  double*d_tau; CUDA_CHECK(cudaMalloc(&d_tau,sizeof(double)*n));
  int gws=0,ows=0;
  CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cs,n,n,d_B,n,&gws));
  CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cs,n,n,n,d_B,n,d_tau,&ows));
  int ws=std::max(gws,ows); double*d_w; CUDA_CHECK(cudaMalloc(&d_w,sizeof(double)*ws));
  int*d_i; CUDA_CHECK(cudaMalloc(&d_i,sizeof(int)));
  CUSOLVER_CHECK(cusolverDnDgeqrf(cs,n,n,d_B,n,d_tau,d_w,gws,d_i));
  CUSOLVER_CHECK(cusolverDnDorgqr(cs,n,n,n,d_B,n,d_tau,d_w,ows,d_i));
  CUDA_CHECK(cudaFree(d_tau)); CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_i));
  {
    std::vector<double> eigs(n); double ic=1.0/cond;
    switch(dist){
      case DIST_GEOMETRIC:
        for(int k=0;k<n;k++) eigs[k]=std::pow(cond,-(double)k/(n-1)); break;
      case DIST_ARITHMETIC:
        for(int k=0;k<n;k++) eigs[k]=1.0-(double)k/(n-1)*(1.0-ic); break;
      case DIST_ONE_FLAT:
        for(int k=0;k<n-1;k++) eigs[k]=1.0; eigs[n-1]=ic; break;
      case DIST_FLAT_ONE:
        eigs[0]=1.0; for(int k=1;k<n;k++) eigs[k]=ic; break;
      case DIST_RANDOM:
        srand(123); eigs[0]=1;
        for(int k=1;k<n-1;k++) eigs[k]=ic+((double)rand()/RAND_MAX)*(1-ic);
        eigs[n-1]=ic; break;
      case DIST_HALF_HALF:
        for(int k=0;k<n;k++) eigs[k]=(k<n/2)?1.0:ic; break;
      case DIST_POLY_DECAY:
        for(int k=0;k<n;k++){
          double t=(double)k/(n-1);
          eigs[k]=(1.0-std::pow(t,shape_p))*(1.0-ic)+ic;
        } break;
      case DIST_MULTI_CLUSTER: {
        int nc=std::max(2,num_clusters);
        int per=n/nc, rem=n-per*nc; int idx=0;
        for(int c=0;c<nc;c++){
          double val=std::pow(cond,-(double)c/(nc-1));
          int cnt=per+(c<rem?1:0);
          for(int j=0;j<cnt&&idx<n;j++) eigs[idx++]=val;
        }
      } break;
      case DIST_POWER_LAW:
        for(int k=0;k<n;k++){
          double t=(double)k/(n-1);
          eigs[k]=std::pow(cond,-std::pow(t,shape_p));
        } break;
      default: fprintf(stderr,"Unknown distribution %d\n",(int)dist); exit(1);
    }
    int bl=256,gr=(n+bl-1)/bl;
    for(int k=0;k<n;k++){double s=std::sqrt(eigs[k]); if(std::abs(s-1)>1e-15) scale_col_kernel<<<gr,bl,0,stream>>>(d_B,n,n,k,s);}
    CUDA_CHECK(cudaGetLastError());
  }
  double one=1,zero=0;
  CUBLAS_CHECK(cublasDgemm(cb,CUBLAS_OP_N,CUBLAS_OP_T,n,n,n,&one,d_B,n,d_B,n,&zero,d_A,lda));
  CUDA_CHECK(cudaFree(d_B));
}

// ---------------------------------------------------------------------------
// Static pivot
// ---------------------------------------------------------------------------
std::vector<int> static_pivot(double*d_A,int n,int lda,cudaStream_t s){
  std::vector<double> hd(n); double*dt; CUDA_CHECK(cudaMalloc(&dt,sizeof(double)*n));
  extract_diag(d_A,dt,n,lda,s);
  CUDA_CHECK(cudaMemcpyAsync(hd.data(),dt,sizeof(double)*n,cudaMemcpyDeviceToHost,s));
  CUDA_CHECK(cudaStreamSynchronize(s)); CUDA_CHECK(cudaFree(dt));
  std::vector<int> p(n); std::iota(p.begin(),p.end(),0);
  std::sort(p.begin(),p.end(),[&](int a,int b){return hd[a]>hd[b];});
  int*dp; CUDA_CHECK(cudaMalloc(&dp,sizeof(int)*n));
  CUDA_CHECK(cudaMemcpyAsync(dp,p.data(),sizeof(int)*n,cudaMemcpyHostToDevice,s));
  std::size_t mb=sizeof(double)*(std::size_t)lda*n;
  double*dt2; CUDA_CHECK(cudaMalloc(&dt2,mb));
  sym_perm_gpu(d_A,dt2,dp,n,lda);
  CUDA_CHECK(cudaMemcpyAsync(d_A,dt2,mb,cudaMemcpyDeviceToDevice,s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  CUDA_CHECK(cudaFree(dp)); CUDA_CHECK(cudaFree(dt2));
  return p;
}

// ---------------------------------------------------------------------------
// Custom adaptive slice selection kernel
// ---------------------------------------------------------------------------

//num bits = 7 + 8*(s - 1) = k => s - 1 = (k - 7) / 8 => s = (k - 7) / 8 + 1 = (k + 1) / 8
__global__ void check_update_kernel(const double*d_A, int lda, double*du,
    const double*dor, int ts, int n, int cs, int ce, int*d_max_bits){
  int i = ts + blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= n) return;
  double sq = 0;
  for(int k = cs; k < ce; k++){
    double v = d_A[(std::size_t)k*lda+i];
    sq += v*v;
  }
  du[i] -= sq;

  int bits;
  if(du[i] <= 0 || dor[i] <= 0){
    bits = 999;  // sentinel: means "use native FP64"
  } else {
    double r = dor[i] / du[i];
    double l = log2(r);
    double sm = 1.0 + (46 - l) / 8;
    int slices = (int)ceil(sm);
    slices = max(3, min(18, slices));
    bits = slices * 8 - 1;
  }
  atomicMax(d_max_bits, bits);
}

// ---------------------------------------------------------------------------
// Synchronous adaptive analysis:  launches the kernel, waits, returns bits.
// ---------------------------------------------------------------------------
int check_update_adaptive(const double*d_A, int lda, double*du, const double*dor,
    int ts, int n, int cs, int ce, int*d_max_bits, cudaStream_t s){
  int cnt = n - ts;
  if(cnt <= 0) return slices_to_bits(3);

  int init_val = 0;
  CUDA_CHECK(cudaMemcpyAsync(d_max_bits, &init_val, sizeof(int), cudaMemcpyHostToDevice, s));
  check_update_kernel<<<(cnt+255)/256, 256, 0, s>>>(d_A, lda, du, dor, ts, n, cs, ce, d_max_bits);
  CUDA_CHECK(cudaGetLastError());
  int h_bits;
  CUDA_CHECK(cudaMemcpyAsync(&h_bits, d_max_bits, sizeof(int), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  if(h_bits >= 999) return 0;
  h_bits = std::max(slices_to_bits(3), std::min(slices_to_bits(18), h_bits));
  return h_bits;
}

// ---------------------------------------------------------------------------
// Error check:  max_{i,j} |A(i,j)-(LL^T)(i,j)| / sqrt(A_ii * A_jj) / n
// ---------------------------------------------------------------------------
__global__ void max_ew_err_kernel(const double*LLT,const double*A,const double* diag_A,
    std::size_t total,int n,int ld,double*bmax){
  extern __shared__ double sd[];
  int t=threadIdx.x; std::size_t g=(std::size_t)blockIdx.x*blockDim.x+t;
  std::size_t st=(std::size_t)gridDim.x*blockDim.x;
  double lm=0;
  for(std::size_t i=g;i<total;i+=st){
    std::size_t c=i/n,r=i%n,m=c*ld+r;
    double ac = A[c*ld + c]; double ar = A[r*ld + r];
    double a=A[m]; if(a!=0) lm=fmax(lm,fabs(LLT[m]-a)/fabs(sqrt(ac*ar)));
  }
  sd[t]=lm; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){if(t<s)sd[t]=fmax(sd[t],sd[t+s]);__syncthreads();}
  if(t==0) bmax[blockIdx.x]=sd[0];
}
double check_fact_gpu(const double*d_L,const double*d_Ar,int n,int lda,cublasHandle_t cb){
  std::size_t mb=sizeof(double)*(std::size_t)lda*n;
  double*dLc; CUDA_CHECK(cudaMalloc(&dLc,mb));
  CUDA_CHECK(cudaMemcpy(dLc,d_L,mb,cudaMemcpyDeviceToDevice));
  zero_upper(dLc,n,lda);
  double*dLLT; CUDA_CHECK(cudaMalloc(&dLLT,mb));
  double one=1,zero=0;
  CUBLAS_CHECK(cublasDgemm(cb,CUBLAS_OP_N,CUBLAS_OP_T,n,n,n,&one,dLc,lda,dLc,lda,&zero,dLLT,lda));
  std::size_t tot=(std::size_t)n*n; int bs=256,nb=std::min(2048,(int)((tot+bs-1)/bs));
  double*dbm; CUDA_CHECK(cudaMalloc(&dbm,sizeof(double)*nb));
  double* diag_A_check; CUDA_CHECK(cudaMalloc(&diag_A_check,sizeof(double)*n));
  extract_diag(d_Ar,diag_A_check,n,lda,0);
  max_ew_err_kernel<<<nb,bs,bs*sizeof(double)>>>(dLLT,d_Ar,diag_A_check,tot,n,lda,dbm);
  CUDA_CHECK(cudaGetLastError());
  std::vector<double> hm(nb);
  CUDA_CHECK(cudaMemcpy(hm.data(),dbm,sizeof(double)*nb,cudaMemcpyDeviceToHost));
  double mx=0; for(int i=0;i<nb;i++) mx=std::max(mx,hm[i]);
  CUDA_CHECK(cudaFree(dLc)); CUDA_CHECK(cudaFree(dLLT)); CUDA_CHECK(cudaFree(dbm));
  CUDA_CHECK(cudaFree(diag_A_check));
  return mx/n;
}

// ===========================================================================
// Per-operation timing accumulator
// ===========================================================================
struct CholTiming {
  float potrf_ms    = 0.0f;
  float trsm_ms     = 0.0f;
  float gemm_ms     = 0.0f;
  float analysis_ms = 0.0f;
  float pivot_ms    = 0.0f;
  float other_ms    = 0.0f;
  int   num_steps   = 0;
};

void print_timing_breakdown(const char* label, const CholTiming& t, float total_ms) {
  float sum = t.potrf_ms + t.trsm_ms + t.gemm_ms + t.analysis_ms + t.pivot_ms + t.other_ms;
  printf("    %s timing breakdown (%d steps, %.3f ms instrumented / %.3f ms total):\n",
         label, t.num_steps, sum, total_ms);
  printf("      POTRF:    %8.3f ms  (%5.1f%%)\n", t.potrf_ms,    100.0*t.potrf_ms/sum);
  printf("      TRSM:     %8.3f ms  (%5.1f%%)\n", t.trsm_ms,     100.0*t.trsm_ms/sum);
  printf("      GEMM:     %8.3f ms  (%5.1f%%)\n", t.gemm_ms,     100.0*t.gemm_ms/sum);
  if(t.analysis_ms > 0)
    printf("      Analysis: %8.3f ms  (%5.1f%%)\n", t.analysis_ms, 100.0*t.analysis_ms/sum);
  if(t.pivot_ms > 0)
    printf("      Pivot:    %8.3f ms  (%5.1f%%)\n", t.pivot_ms,    100.0*t.pivot_ms/sum);
  if(t.other_ms > 0)
    printf("      Other:    %8.3f ms  (%5.1f%%)\n", t.other_ms,    100.0*t.other_ms/sum);
}

#define CHOL_EV_START(ev0, stream, timing) \
  do { if(timing) { CUDA_CHECK(cudaEventRecord(ev0, stream)); } } while(0)

#define CHOL_EV_STOP(ev0, ev1, stream, timing, field) \
  do { if(timing) { \
    CUDA_CHECK(cudaEventRecord(ev1, stream)); \
    CUDA_CHECK(cudaEventSynchronize(ev1)); \
    float _ms; CUDA_CHECK(cudaEventElapsedTime(&_ms, ev0, ev1)); \
    timing->field += _ms; \
  } } while(0)

// ===========================================================================
// Cholesky Variant 1: cuBLAS native FP64 (baseline) — RIGHT-LOOKING
// ===========================================================================
void chol_cublas_native(double*d_A,int n,int lda,int nb,
    cublasHandle_t cb,cusolverDnHandle_t cs,double*pw,int pws,int*di,
    double*dor,double*du,cudaStream_t s,
    pivot_mode_t pm,std::vector<int>&gp,
    CholTiming*timing=nullptr){
  cusolverMathMode_t orig_mode;
  cusolverDnGetMathMode(cs, &orig_mode);
  CUSOLVER_CHECK(cusolverDnDpotrf(cs, CUBLAS_FILL_MODE_LOWER,
                 n, d_A, lda, pw, pws, di));
  cusolverDnSetMathMode(cs, orig_mode);
}

// ===========================================================================
// Cholesky Variant 2: cuBLAS Ozaki fixed slices — RIGHT-LOOKING
// ===========================================================================
void chol_cublas_fixed(double*d_A,int n,int lda,int nb,int fixed_bits,
    cublasHandle_t cb_emu, cublasHandle_t cb_native, cusolverDnHandle_t cs,double*pw,int pws,int*di,
    double*dor,double*du,cudaStream_t s,
    pivot_mode_t pm,std::vector<int>&gp,
    CholTiming*timing=nullptr){
  double an=-1,one=1;
  for(int j=0;j<n;j+=nb){
    int jb=std::min(nb,n-j);
    double*Ajj=d_A+(std::size_t)j*lda+j;

    if(pm!=PIVOT_STATIC) panel_pivot(d_A,n,lda,dor,du,gp,j,jb,pm,s);

    CUSOLVER_CHECK(cusolverDnDpotrf(cs,CUBLAS_FILL_MODE_LOWER,jb,Ajj,lda,pw,pws,di));

    if(j+jb<n){
      int rem=n-j-jb;
      double*Aij=d_A+(std::size_t)j*lda+(j+jb);

      CUBLAS_CHECK(cublasDtrsm(cb_emu,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,rem,jb,&one,Ajj,lda,Aij,lda));

      double*A22=d_A+(std::size_t)(j+jb)*lda+(j+jb);
      CUBLAS_CHECK(cublasDgemm(cb_emu,CUBLAS_OP_N,CUBLAS_OP_T,
        rem,rem,jb,&an,Aij,lda,Aij,lda,&one,A22,lda));

      if(pm!=PIVOT_STATIC)
        extract_diag(d_A,du,n,lda,s);
    }
  }
}

// ===========================================================================
// Cholesky Variant 3: cuBLAS ADP (dynamic mantissa control) — RIGHT-LOOKING
// ===========================================================================
struct AdpRecord {
  int mantissa_bits;
  int slices;
  bool native_fp64;
};

void chol_cublas_adp(double*d_A,int n,int lda,int nb,
    cublasHandle_t cb_adp, cublasHandle_t cb_native, cusolverDnHandle_t cs,double*pw,int pws,int*di,
    double*dor,double*du,int*d_adp_bits,cudaStream_t s,
    pivot_mode_t pm,std::vector<int>&gp,
    std::vector<AdpRecord>*history,
    CholTiming*timing=nullptr){
  double an=-1,one=1;

  auto read_adp = [&](){
    if(!history) return;
    int h_bits;
    CUDA_CHECK(cudaMemcpyAsync(&h_bits, d_adp_bits, sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    AdpRecord rec;
    rec.mantissa_bits = h_bits;
    rec.native_fp64 = (h_bits == 0);
    rec.slices = (h_bits > 0) ? bits_to_slices(h_bits) : 0;
    history->push_back(rec);
  };

  for(int j=0;j<n;j+=nb){
    int jb=std::min(nb,n-j);
    double*Ajj=d_A+(std::size_t)j*lda+j;

    if(pm!=PIVOT_STATIC) panel_pivot(d_A,n,lda,dor,du,gp,j,jb,pm,s);

    CUSOLVER_CHECK(cusolverDnDpotrf(cs,CUBLAS_FILL_MODE_LOWER,jb,Ajj,lda,pw,pws,di));

    if(j+jb<n){
      int rem=n-j-jb;
      double*Aij=d_A+(std::size_t)j*lda+(j+jb);

      CUBLAS_CHECK(cublasDtrsm(cb_adp,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,rem,jb,&one,Ajj,lda,Aij,lda));

      int sentinel = -1;
      CUDA_CHECK(cudaMemcpyAsync(d_adp_bits, &sentinel, sizeof(int), cudaMemcpyHostToDevice, s));

      double*A22=d_A+(std::size_t)(j+jb)*lda+(j+jb);
      CUBLAS_CHECK(cublasDgemm(cb_adp,CUBLAS_OP_N,CUBLAS_OP_T,
        rem,rem,jb,&an,Aij,lda,Aij,lda,&one,A22,lda));
      read_adp();

      if(pm!=PIVOT_STATIC)
        extract_diag(d_A,du,n,lda,s);
    }
  }
}

// ===========================================================================
// Cholesky Variant 4: Custom adaptive — RIGHT-LOOKING
// ===========================================================================
// Assumed CholTiming struct — add these fields if not present:


// Helper: create a pair of events, record start on stream, return them
// Usage:  auto [ev_start, ev_stop] = make_event_pair(stream);
//         ... kernel ...
//         float ms = stop_event_pair(ev_start, ev_stop, stream);
//#define PROFILE
#ifdef PROFILE
#  define IF_PROFILE(x) x
#else
#  define IF_PROFILE(x)
#endif
static std::pair<cudaEvent_t,cudaEvent_t> make_event_pair() {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    return {a, b};
}
static float stop_event_pair(cudaEvent_t a, cudaEvent_t b,
                              cudaStream_t stream) {
    cudaEventRecord(b, stream);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms;
}void chol_cublas_custom_adaptive(double*d_A,int n,int lda,int nb,
    cublasHandle_t cb_adap,
    cublasHandle_t cb_native,
    cusolverDnHandle_t cs,double*pw,int pws,int*di,
    double*dor,double*du,int*d_max_bits,
    cudaStream_t s,
    cudaStream_t analysis_stream,
    cudaEvent_t  analysis_done,
    pivot_mode_t pm,std::vector<int>&gp,
    std::vector<int>*slice_history,
    CholTiming*timing=nullptr)
{
    double an=-1, one=1;

#ifdef PROFILE
    bool timing_owned = false;
    if(!timing){
        timing      = new CholTiming{};
        timing_owned = true;
    }

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    auto time_ms = [&](auto fn) -> float {
        cudaEventRecord(ev0, s);
        fn();
        cudaEventRecord(ev1, s);
        cudaEventSynchronize(ev1);
        float ms=0; cudaEventElapsedTime(&ms, ev0, ev1);
        return ms;
    };
#else
    auto time_ms = [&](auto fn) -> float { fn(); return 0.f; };
#endif

    for(int j=0;j<n;j+=nb){
        int jb=std::min(nb,n-j);
        double*Ajj=d_A+(std::size_t)j*lda+j;

        if(pm!=PIVOT_STATIC)
            IF_PROFILE(timing->pivot_ms +=) time_ms([&]{
                panel_pivot(d_A,n,lda,dor,du,gp,j,jb,pm,s);
            });

        IF_PROFILE(timing->potrf_ms +=) time_ms([&]{
            CUSOLVER_CHECK(cusolverDnDpotrf(cs,CUBLAS_FILL_MODE_LOWER,
                                            jb,Ajj,lda,pw,pws,di));
        });

        if(j+jb<n){
            int rem=n-j-jb;
            double*Aij=d_A+(std::size_t)j*lda+(j+jb);

            IF_PROFILE(timing->trsm_ms +=) time_ms([&]{
                CUBLAS_CHECK(cublasDtrsm(cb_adap,
                    CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,
                    rem,jb,&one,Ajj,lda,Aij,lda));
            });

            int cur_bits;
            IF_PROFILE(timing->analysis_ms +=) time_ms([&]{
                cur_bits = check_update_adaptive(d_A,lda,du,dor,
                               j+jb,n,j,j+jb,d_max_bits,s);
            });

            if(slice_history)
                slice_history->push_back(cur_bits==0 ? 0 : bits_to_slices(cur_bits));

            double*A22=d_A+(std::size_t)(j+jb)*lda+(j+jb);

            IF_PROFILE(timing->gemm_ms +=) time_ms([&]{
                if(cur_bits==0){
                    CUBLAS_CHECK(cublasDgemm(cb_native,CUBLAS_OP_N,CUBLAS_OP_T,
                        rem,rem,jb,&an,Aij,lda,Aij,lda,&one,A22,lda));
                } else {
                    CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(
                                     cb_adap,cur_bits));
                    CUBLAS_CHECK(cublasDgemm(cb_adap,CUBLAS_OP_N,CUBLAS_OP_T,
                        rem,rem,jb,&an,Aij,lda,Aij,lda,&one,A22,lda));
                }
            });
        }

#ifdef PROFILE
        timing->num_steps++;
#endif
    }

#ifdef PROFILE
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    float tot = timing->potrf_ms + timing->trsm_ms + timing->gemm_ms
              + timing->analysis_ms + timing->pivot_ms + timing->other_ms;
    auto pct = [&](float v){ return tot>0 ? 100.f*v/tot : 0.f; };
    printf("\n=== CholTiming  n=%-5d  nb=%-4d  steps=%d ===\n",
           n, nb, timing->num_steps);
    printf("  potrf      %8.2f ms  %5.1f%%\n", timing->potrf_ms,    pct(timing->potrf_ms));
    printf("  trsm       %8.2f ms  %5.1f%%\n", timing->trsm_ms,     pct(timing->trsm_ms));
    printf("  gemm       %8.2f ms  %5.1f%%\n", timing->gemm_ms,     pct(timing->gemm_ms));
    printf("  analysis   %8.2f ms  %5.1f%%\n", timing->analysis_ms, pct(timing->analysis_ms));
    printf("  pivot      %8.2f ms  %5.1f%%\n", timing->pivot_ms,    pct(timing->pivot_ms));
    printf("  other      %8.2f ms  %5.1f%%\n", timing->other_ms,    pct(timing->other_ms));
    printf("  ──────────────────────────────────\n");
    printf("  total      %8.2f ms\n", tot);

    if(timing_owned){
        delete timing;
        timing = nullptr;
    }
#endif
}
// ===========================================================================
// Helpers
// ===========================================================================
void prepare(double*d_A,const double*d_src,double*dor,double*du,const double*dos,
             std::vector<int>&gp,const std::vector<int>&bp,int n,std::size_t mb,cudaStream_t s){
  CUDA_CHECK(cudaMemcpyAsync(d_A,d_src,mb,cudaMemcpyDeviceToDevice,s));
  CUDA_CHECK(cudaMemcpyAsync(dor,dos,sizeof(double)*n,cudaMemcpyDeviceToDevice,s));
  CUDA_CHECK(cudaMemcpyAsync(du,dos,sizeof(double)*n,cudaMemcpyDeviceToDevice,s));
  gp=bp; CUDA_CHECK(cudaStreamSynchronize(s));
}

void build_ref(const double*d_A_orig,double*d_ref,const std::vector<int>&gp,
               int n,int lda,cudaStream_t s){
  int*dp; CUDA_CHECK(cudaMalloc(&dp,sizeof(int)*n));
  CUDA_CHECK(cudaMemcpy(dp,gp.data(),sizeof(int)*n,cudaMemcpyHostToDevice));
  sym_perm_gpu(d_A_orig,d_ref,dp,n,lda);
  CUDA_CHECK(cudaStreamSynchronize(s));
  CUDA_CHECK(cudaFree(dp));
}

// ===========================================================================
// Print slice history comparison table
// ===========================================================================
void print_slice_table(const std::vector<AdpRecord>& adp_hist,
                       const std::vector<int>& adapt_hist) {
  size_t nsteps = std::max(adp_hist.size(), adapt_hist.size());
  if(nsteps == 0) return;

  printf("\n  %-6s | %-22s | %-22s\n", "Step", "cuBLAS ADP", "Custom Adaptive");
  printf("  %-6s | %-10s %-10s | %-10s %-10s\n",
         "","Slices","(bits)", "Slices","(bits)");
  printf("  -------+------------------------+------------------------\n");

  int adp_total = 0, adp_native_count = 0;
  int adapt_total = 0, adapt_native_count = 0;

  for(size_t i = 0; i < nsteps; i++){
    printf("  %4zu   |", i+1);

    if(i < adp_hist.size()){
      const auto& r = adp_hist[i];
      if(r.native_fp64 || r.mantissa_bits <= 0){
        printf(" %-10s %-10s |", "FP64", "--");
        adp_native_count++;
      } else {
        printf(" %-10d %-10d |", r.slices, r.mantissa_bits);
        adp_total += r.slices;
      }
    } else {
      printf(" %-10s %-10s |", "--", "--");
    }

    if(i < adapt_hist.size()){
      int sl = adapt_hist[i];
      if(sl == 0){
        printf(" %-10s %-10s", "FP64", "--");
        adapt_native_count++;
      } else {
        printf(" %-10d %-10d", sl, slices_to_bits(sl));
        adapt_total += sl;
      }
    } else {
      printf(" %-10s %-10s", "--", "--");
    }
    printf("\n");
  }

  printf("  -------+------------------------+------------------------\n");

  int adp_emu_steps = (int)adp_hist.size() - adp_native_count;
  int adapt_emu_steps = (int)adapt_hist.size() - adapt_native_count;

  printf("  Summary:\n");
  if(!adp_hist.empty()){
    printf("    ADP:      %d steps, %d emulated (avg %.1f slices), %d native FP64\n",
           (int)adp_hist.size(), adp_emu_steps,
           adp_emu_steps > 0 ? (double)adp_total / adp_emu_steps : 0.0,
           adp_native_count);
  }
  if(!adapt_hist.empty()){
    printf("    Adaptive: %d steps, %d emulated (avg %.1f slices), %d native FP64\n",
           (int)adapt_hist.size(), adapt_emu_steps,
           adapt_emu_steps > 0 ? (double)adapt_total / adapt_emu_steps : 0.0,
           adapt_native_count);
  }

  auto print_dist = [](const char* label, auto get_slices, size_t count) {
    int hist[20] = {};
    int native = 0;
    for(size_t i = 0; i < count; i++){
      int s = get_slices(i);
      if(s == 0) native++;
      else if(s >= 1 && s <= 18) hist[s]++;
    }
    printf("    %s distribution: ", label);
    if(native > 0) printf("FP64:%d ", native);
    for(int s = 3; s <= 18; s++) if(hist[s]) printf("%d-sl:%d ", s, hist[s]);
    printf("\n");
  };

  if(!adp_hist.empty()){
    print_dist("ADP",
      [&](size_t i) -> int { return adp_hist[i].native_fp64 ? 0 : adp_hist[i].slices; },
      adp_hist.size());
  }
  if(!adapt_hist.empty()){
    print_dist("Adaptive",
      [&](size_t i) -> int { return adapt_hist[i]; },
      adapt_hist.size());
  }
}

// ===========================================================================
// Benchmark helpers
// ===========================================================================
static const int NWARMUP = 3;
static const int NRUNS   = 5;

float median_of(std::vector<float>& v) {
  std::sort(v.begin(), v.end());
  int n = (int)v.size();
  if (n % 2 == 1) return v[n/2];
  return 0.5f * (v[n/2 - 1] + v[n/2]);
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char **argv) {
  int n       = (argc>1)?atoi(argv[1]):4096;
  int nb      = (argc>2)?atoi(argv[2]):256;
  double cond = (argc>3)?atof(argv[3]):1e6;
  int dist_id = (argc>4)?atoi(argv[4]):1;
  int piv_id  = (argc>5)?atoi(argv[5]):0;
  int equil   = (argc>6)?atoi(argv[6]):0;
  double shape_p     = (argc>7)?atof(argv[7]):2.0;
  int num_clusters   = (argc>8)?atoi(argv[8]):4;
  const char* csv_path = (argc>9)?argv[9]:nullptr;
  const char* mtx_path = (argc>10)?argv[10]:nullptr;
  int mtx_max_dim      = (argc>11)?atoi(argv[11]):40000;
  int stress_test = (argc>12)?atoi(argv[12]):0; //tests for emulation vs floating point

  // Fixed-slice variant uses 7 slices = 55 mantissa bits.
  int fixed_slices = 7;
  int fixed_bits   = slices_to_bits(fixed_slices);
  char fixed_variant_name[32];
  snprintf(fixed_variant_name, sizeof(fixed_variant_name), "fixed_%d", fixed_slices);

  bool use_mtx = (mtx_path != nullptr);

  if (!use_mtx) {
    if(dist_id<1||dist_id>9){fprintf(stderr,"dist 1-9\n");return 1;}
  }
  if(piv_id<0||piv_id>2){fprintf(stderr,"pivot 0-2\n");return 1;}
  auto dist=static_cast<eigenvalue_dist_t>(dist_id);
  auto pivot=static_cast<pivot_mode_t>(piv_id);

  // -- CUDA infrastructure that does NOT depend on n.  We need cuSOLVER up
  //    front so we can run the SPD probe on a freshly-loaded MTX matrix
  //    before committing to the rest of the workspace allocations.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaStream_t analysis_stream;
  CUDA_CHECK(cudaStreamCreate(&analysis_stream));
  cudaEvent_t analysis_done;
  CUDA_CHECK(cudaEventCreateWithFlags(&analysis_done, cudaEventDisableTiming));
  cusolverDnHandle_t cs;
  CUSOLVER_CHECK(cusolverDnCreate(&cs));
  CUSOLVER_CHECK(cusolverDnSetStream(cs, stream));

  // -- If an MTX file was given, load + verify SPD + pad.  This may change n.
  double* d_loaded = nullptr;
  std::string dist_label_storage;
  const char* dist_label = nullptr;

  if (use_mtx) {
    printf("================================================================\n");
    printf("  Loading matrix from Matrix Market file\n");
    printf("================================================================\n");
    if (load_mtx_to_dense_f64(mtx_path, &d_loaded, &n, mtx_max_dim) != 0) {
      fprintf(stderr, "  Failed to load MTX file %s\n", mtx_path);
      cusolverDnDestroy(cs);
      cudaStreamDestroy(stream);
      cudaStreamDestroy(analysis_stream);
      cudaEventDestroy(analysis_done);
      return 1;
    }

    if (!check_spd_gpu(d_loaded, n, n, cs, stream)) {
      fprintf(stderr, "  Matrix from %s is not SPD; aborting.\n", mtx_path);
      CUDA_CHECK(cudaFree(d_loaded));
      cusolverDnDestroy(cs);
      cudaStreamDestroy(stream);
      cudaStreamDestroy(analysis_stream);
      cudaEventDestroy(analysis_done);
      return 1;
    }

    pad_spd_to_multiple(&d_loaded, &n, nb);

    dist_label_storage = "mtx:" + filename_stem(mtx_path);
    dist_label = dist_label_storage.c_str();

    // Override sweep-style metadata with neutral values for CSV recording.
    cond = 0.0;
    shape_p = 0.0;
    num_clusters = 0;
  } else {
    dist_label = dist_name(dist);
  }

  printf("================================================================\n");
  printf("  RIGHT-LOOKING Cholesky (POTRF -> TRSM -> GEMM)\n");
  if (use_mtx) {
    printf("  Source: %s (padded to multiple of nb=%d)\n", mtx_path, nb);
  }
  // Standardized header line — log parser anchors on this format for
  // associating per-step slice tables with a configuration, so we print it
  // for both synthetic and MTX runs.
  printf("  n=%d nb=%d cond=%.1e dist=%s pivot=%s equil=%s\n",
         n, nb, cond, dist_label, pivot_name(pivot), equil?"compare":"off");
  if (!use_mtx) {
    if(dist==DIST_POLY_DECAY||dist==DIST_POWER_LAW)
      printf("  shape_p=%.2f\n", shape_p);
    if(dist==DIST_MULTI_CLUSTER)
      printf("  num_clusters=%d\n", num_clusters);
  }
  printf("  Fixed slices=%d (%d mantissa bits)\n", fixed_slices, fixed_bits);
  printf("  Benchmark: %d warmup + %d timed runs (median)\n", NWARMUP, NRUNS);
  if(csv_path) printf("  CSV output: %s\n", csv_path);
  printf("================================================================\n\n");

  FILE* csv_fp = nullptr;
  if(csv_path){
    bool write_header = false;
    FILE* test = fopen(csv_path, "r");
    if(!test){ write_header = true; }
    else { fseek(test,0,SEEK_END); if(ftell(test)==0) write_header=true; fclose(test); }
    csv_fp = fopen(csv_path, "a");
    if(!csv_fp){fprintf(stderr,"Cannot open CSV %s\n",csv_path);return 1;}
    if(write_header){
      fprintf(csv_fp, "n,nb,cond,dist,pivot,equil,shape_p,num_clusters,"
              "variant,time_ms,bwd_err,avg_slices,native_fp64_steps,total_steps\n");
    }
  }

  std::size_t mb = sizeof(double)*(std::size_t)n*n;

  // -- cuBLAS handles --
  cublasHandle_t cb_native;
  CUBLAS_CHECK(cublasCreate(&cb_native));
  CUBLAS_CHECK(cublasSetStream(cb_native, stream));

  cublasHandle_t cb_fixed;
  CUBLAS_CHECK(cublasCreate(&cb_fixed));
  CUBLAS_CHECK(cublasSetStream(cb_fixed, stream));
  CUBLAS_CHECK(cublasSetMathMode(cb_fixed, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
  CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(cb_fixed,
      CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
  CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(cb_fixed, fixed_bits));

  cublasHandle_t cb_adp;
  CUBLAS_CHECK(cublasCreate(&cb_adp));
  CUBLAS_CHECK(cublasSetStream(cb_adp, stream));
  CUBLAS_CHECK(cublasSetMathMode(cb_adp, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
  CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(cb_adp,
      CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));
  int* d_adp_bits;
  CUDA_CHECK(cudaMalloc(&d_adp_bits, sizeof(int)));
  CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaBitCountPointer(cb_adp, d_adp_bits));

  cublasHandle_t cb_adap;
  CUBLAS_CHECK(cublasCreate(&cb_adap));
  CUBLAS_CHECK(cublasSetStream(cb_adap, stream));
  CUBLAS_CHECK(cublasSetMathMode(cb_adap, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
  CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(cb_adap,
      CUDA_EMULATION_MANTISSA_CONTROL_FIXED));

  // -- FP-emulation workspace --
  void* d_fp_workspace = nullptr;
  size_t fp_ws_bytes = 0;
  {
    const int ws_bits = 80;
    int nsl = (ws_bits + 1 + 7) / 8;
    int pm = ((n+1023)/1024)*1024;
    int pk = ((nb+127)/128)*128;
    int nbk = ((nb+63)/64);
    size_t gem = (size_t)((size_t)pm*pk + (size_t)pm*pk) * nsl
               + (size_t)4*((size_t)pm + pm);
    size_t adp = (size_t)4*((size_t)n*nbk + (size_t)n*nbk + (size_t)n*n);
    fp_ws_bytes = (size_t)(std::max(gem, adp) * 1.25) + 128ULL*1024*1024;
    CUDA_CHECK(cudaMalloc(&d_fp_workspace, fp_ws_bytes));
    CUBLAS_CHECK(cublasSetWorkspace(cb_fixed, d_fp_workspace, fp_ws_bytes));
    CUBLAS_CHECK(cublasSetWorkspace(cb_adp,   d_fp_workspace, fp_ws_bytes));
    CUBLAS_CHECK(cublasSetWorkspace(cb_adap,  d_fp_workspace, fp_ws_bytes));
    printf("  FP emulation workspace: %.1f MB (80 mantissa bits, %d slices)\n",
           fp_ws_bytes / (1024.0*1024.0), nsl);
  }

  // -- Per-n allocations --
  double*d_A; CUDA_CHECK(cudaMalloc(&d_A,mb));
  int pws=0;
  CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(cs,CUBLAS_FILL_MODE_LOWER,n,d_A,n,&pws));
  double*d_pw; CUDA_CHECK(cudaMalloc(&d_pw,sizeof(double)*pws));
  int*d_info; CUDA_CHECK(cudaMalloc(&d_info,sizeof(int)));
  double *d_Ao = nullptr, *d_Ap = nullptr, *d_Ar = nullptr;
  CUDA_CHECK(cudaMalloc(&d_Ap,mb));
  CUDA_CHECK(cudaMalloc(&d_Ar,mb));
  double *dor, *du, *dos, *d_scale;
  CUDA_CHECK(cudaMalloc(&dor,sizeof(double)*n));
  CUDA_CHECK(cudaMalloc(&du, sizeof(double)*n));
  CUDA_CHECK(cudaMalloc(&dos,sizeof(double)*n));
  CUDA_CHECK(cudaMalloc(&d_scale,sizeof(double)*n));
  int*d_max_bits; CUDA_CHECK(cudaMalloc(&d_max_bits,sizeof(int)));

  // -- Source matrix:  either reuse the loaded MTX buffer or generate on GPU
  if (d_loaded) {
    d_Ao = d_loaded;       // takes ownership
    d_loaded = nullptr;
    printf("  Using matrix loaded from %s.\n", mtx_path);
  } else {
    CUDA_CHECK(cudaMalloc(&d_Ao, mb));
    printf("Generating matrix ... "); fflush(stdout);
    generate_spd(d_Ao,n,n,cond,dist,cb_native,cs,stream,shape_p,num_clusters);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("done.\n");
  }

  if (n <= 16) {
    double* h_Ao = new double[n * n];
    CUDA_CHECK(cudaMemcpy(h_Ao, d_Ao, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
    print_Matrix(h_Ao, n, n);
    delete[] h_Ao;
  }

  cudaEvent_t t0,t1;
  CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

  const int num_equil = equil ? 2 : 1;

  for (int eq = 0; eq < num_equil; eq++) {
    bool do_eq = (eq == 1);
    printf("\n================================================================\n");
    printf("  %s\n", do_eq ? "WITH equilibration (power-of-2)" : "WITHOUT equilibration");
    printf("================================================================\n");

    CUDA_CHECK(cudaMemcpy(d_Ap,d_Ao,mb,cudaMemcpyDeviceToDevice));
    if (do_eq) {
      printf("  Equilibrating ... "); fflush(stdout);
      equilibrate_gpu(d_Ap, n, n, d_scale, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      printf("done.\n");
    }

    printf("  Static pivoting ... "); fflush(stdout);
    std::vector<int> base_perm = static_pivot(d_Ap, n, n, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("done.\n");

    double *d_scale_perm = nullptr;
    if (do_eq) {
      CUDA_CHECK(cudaMalloc(&d_scale_perm, sizeof(double) * n));
      int *d_bp; CUDA_CHECK(cudaMalloc(&d_bp, sizeof(int) * n));
      CUDA_CHECK(cudaMemcpy(d_bp, base_perm.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
      permute_vec_kernel<<<(n+255)/256, 256, 0, stream>>>(d_scale, d_scale_perm, d_bp, n);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaFree(d_bp));
    }

    extract_diag(d_Ap, dos, n, n, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int> gp;
    float elapsed;

    auto finish_and_check = [&](const char* /*label*/) -> double {
      if (do_eq) {
        double *d_sf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sf, sizeof(double) * n));
        int *d_gp; CUDA_CHECK(cudaMalloc(&d_gp, sizeof(int) * n));
        CUDA_CHECK(cudaMemcpy(d_gp, gp.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
        permute_vec_kernel<<<(n+255)/256, 256, 0, stream>>>(d_scale, d_sf, d_gp, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        unequilibrate_L(d_A, d_sf, n, n, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_sf));
        CUDA_CHECK(cudaFree(d_gp));
      }
      build_ref(d_Ao, d_Ar, gp, n, n, stream);
      return check_fact_gpu(d_A, d_Ar, n, n, cb_native);
    };

    // =====================================================================
    // (1) cuBLAS native FP64 baseline
    // =====================================================================
    for(int w=0;w<NWARMUP;w++){
      prepare(d_A,d_Ap,dor,du,dos,gp,base_perm,n,mb,stream);
      chol_cublas_native(d_A,n,n,nb,cb_native,cs,d_pw,pws,d_info,dor,du,stream,pivot,gp);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::vector<float> times_native(NRUNS);
    for(int r=0;r<NRUNS;r++){
      prepare(d_A,d_Ap,dor,du,dos,gp,base_perm,n,mb,stream);
      CUDA_CHECK(cudaEventRecord(t0,stream));
      chol_cublas_native(d_A,n,n,nb,cb_native,cs,d_pw,pws,d_info,dor,du,stream,pivot,gp);
      CUDA_CHECK(cudaEventRecord(t1,stream)); CUDA_CHECK(cudaEventSynchronize(t1));
      CUDA_CHECK(cudaEventElapsedTime(&times_native[r],t0,t1));
    }
    elapsed = median_of(times_native);
    double ec = finish_and_check("native");
    printf("  [cuBLAS native FP64]  %8.3f ms   bwd_err = %e\n", elapsed, ec);
    float tc = elapsed;

    // =====================================================================
    // (2) cuBLAS Ozaki fixed slices
    // =====================================================================
    for(int w=0;w<NWARMUP;w++){
      prepare(d_A,d_Ap,dor,du,dos,gp,base_perm,n,mb,stream);
      chol_cublas_fixed(d_A,n,n,nb,fixed_bits,cb_fixed, cb_native, cs, d_pw,pws,d_info,dor,du,stream,pivot,gp);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::vector<float> times_fixed(NRUNS);
    for(int r=0;r<NRUNS;r++){
      prepare(d_A,d_Ap,dor,du,dos,gp,base_perm,n,mb,stream);
      CUDA_CHECK(cudaEventRecord(t0,stream));
      chol_cublas_fixed(d_A,n,n,nb,fixed_bits,cb_fixed, cb_native,cs,d_pw,pws,d_info,dor,du,stream,pivot,gp);
      CUDA_CHECK(cudaEventRecord(t1,stream)); CUDA_CHECK(cudaEventSynchronize(t1));
      CUDA_CHECK(cudaEventElapsedTime(&times_fixed[r],t0,t1));
    }
    elapsed = median_of(times_fixed);
    double ef = finish_and_check("fixed");
    printf("  [Ozaki fix-%d]         %8.3f ms   bwd_err = %e\n", fixed_slices, elapsed, ef);
    float tf = elapsed;

    // =====================================================================
    // (3) cuBLAS ADP (dynamic)
    // =====================================================================
    std::vector<AdpRecord> adp_hist;
    adp_hist.reserve((n+nb-1)/nb);
    for(int w=0;w<NWARMUP;w++){
      prepare(d_A,d_Ap,dor,du,dos,gp,base_perm,n,mb,stream);
      chol_cublas_adp(d_A,n,n,nb,cb_adp, cb_native, cs,d_pw,pws,d_info,dor,du,d_adp_bits,stream,pivot,gp,nullptr);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::vector<float> times_adp(NRUNS);
    for(int r=0;r<NRUNS;r++){
      adp_hist.clear();
      prepare(d_A,d_Ap,dor,du,dos,gp,base_perm,n,mb,stream);
      CUDA_CHECK(cudaEventRecord(t0,stream));
      chol_cublas_adp(d_A,n,n,nb,cb_adp, cb_native,cs,d_pw,pws,d_info,dor,du,d_adp_bits,stream,pivot,gp,
                      (r==NRUNS-1) ? &adp_hist : nullptr);
      CUDA_CHECK(cudaEventRecord(t1,stream)); CUDA_CHECK(cudaEventSynchronize(t1));
      CUDA_CHECK(cudaEventElapsedTime(&times_adp[r],t0,t1));
    }
    elapsed = median_of(times_adp);
    double ea = finish_and_check("adp");
    printf("  [Ozaki ADP dynamic]   %8.3f ms   bwd_err = %e\n", elapsed, ea);
    float ta = elapsed;

    // =====================================================================
    // (4) Custom adaptive
    // =====================================================================
    std::vector<int> adapt_hist;
    adapt_hist.reserve((n+nb-1)/nb);
    for(int w=0;w<NWARMUP;w++){
      prepare(d_A,d_Ap,dor,du,dos,gp,base_perm,n,mb,stream);
      chol_cublas_custom_adaptive(d_A,n,n,nb,cb_adap,cb_native,cs,d_pw,pws,d_info,
          dor,du,d_max_bits,stream,analysis_stream,analysis_done,pivot,gp,nullptr);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    std::vector<float> times_adapt(NRUNS);
    for(int r=0;r<NRUNS;r++){
      adapt_hist.clear();
      prepare(d_A,d_Ap,dor,du,dos,gp,base_perm,n,mb,stream);
      CUDA_CHECK(cudaEventRecord(t0,stream));
      chol_cublas_custom_adaptive(d_A,n,n,nb,cb_adap,cb_native,cs,d_pw,pws,d_info,
          dor,du,d_max_bits,stream,analysis_stream,analysis_done,pivot,gp,
          (r==NRUNS-1) ? &adapt_hist : nullptr);
      CUDA_CHECK(cudaEventRecord(t1,stream)); CUDA_CHECK(cudaEventSynchronize(t1));
      CUDA_CHECK(cudaEventElapsedTime(&times_adapt[r],t0,t1));
    }
    elapsed = median_of(times_adapt);
    double eadapt = finish_and_check("custom-adaptive");
    printf("  [Custom adaptive]     %8.3f ms   bwd_err = %e\n", elapsed, eadapt);
    float tadapt = elapsed;

    // =====================================================================
    // Per-run timings
    // =====================================================================
    printf("\n  Per-run timings (ms):\n");
    printf("  %6s  %10s %10s %10s %10s\n", "Run", "Native",
           "Fix", "ADP", "Adaptive");
    for(int r=0;r<NRUNS;r++){
      printf("  %6d  %10.3f %10.3f %10.3f %10.3f\n", r+1,
             times_native[r], times_fixed[r], times_adp[r], times_adapt[r]);
    }
    printf("  %6s  %10.3f %10.3f %10.3f %10.3f  (median)\n", ">>",
           tc, tf, ta, tadapt);

    // =====================================================================
    // Internal timing breakdown
    // =====================================================================
    printf("\n  Internal timing breakdown (dedicated instrumented run):\n");
    {
      CholTiming tm_native{};
      prepare(d_A,d_Ap,dor,du,dos,gp,base_perm,n,mb,stream);
      chol_cublas_native(d_A,n,n,nb,cb_native,cs,d_pw,pws,d_info,
          dor,du,stream,pivot,gp,&tm_native);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      CholTiming tm_fixed{};
      prepare(d_A,d_Ap,dor,du,dos,gp,base_perm,n,mb,stream);
      chol_cublas_fixed(d_A,n,n,nb,fixed_bits,cb_fixed, cb_native, cs,d_pw,pws,d_info,
          dor,du,stream,pivot,gp,&tm_fixed);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      CholTiming tm_adp{};
      prepare(d_A,d_Ap,dor,du,dos,gp,base_perm,n,mb,stream);
      chol_cublas_adp(d_A,n,n,nb,cb_adp, cb_native,cs,d_pw,pws,d_info,
          dor,du,d_adp_bits,stream,pivot,gp,nullptr,&tm_adp);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      CholTiming tm_adapt{};
      prepare(d_A,d_Ap,dor,du,dos,gp,base_perm,n,mb,stream);
      chol_cublas_custom_adaptive(d_A,n,n,nb,cb_adap,cb_native,cs,d_pw,pws,d_info,
          dor,du,d_max_bits,stream,analysis_stream,analysis_done,pivot,gp,
          nullptr,&tm_adapt);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // =====================================================================
    // Slice history comparison
    // =====================================================================
    print_slice_table(adp_hist, adapt_hist);

    // =====================================================================
    // Speedup summary
    // =====================================================================
    printf("\n  Speedups vs native FP64 (%.3f ms):\n", tc);
    printf("    fix-%d:          %.2fx\n", fixed_slices, (double)tc/tf);
    printf("    ADP dynamic:    %.2fx\n", (double)tc/ta);
    printf("    custom adaptive:%.2fx\n", (double)tc/tadapt);
    printf("    ADP/fix-%d:      %.2fx\n", fixed_slices, (double)tf/ta);
    printf("    custom/fix-%d:   %.2fx\n", fixed_slices, (double)tf/tadapt);

    // =====================================================================
    // CSV output
    // =====================================================================
    if(csv_fp){
      const char* eq_str = do_eq ? "yes" : "no";
      auto write_row = [&](const char* variant, float time, double err,
                           double avg_sl, int native_count, int total_steps){
        fprintf(csv_fp, "%d,%d,%.2e,%s,%s,%s,%.2f,%d,%s,%.3f,%.6e,%.1f,%d,%d\n",
                n, nb, cond, dist_label, pivot_name(pivot), eq_str,
                shape_p, num_clusters, variant, time, err,
                avg_sl, native_count, total_steps);
      };

      write_row("native_fp64", tc, ec, 0.0, 0, 0);

      int nsteps_f = (n + nb - 1) / nb - 1;
      write_row(fixed_variant_name, tf, ef, (double)fixed_slices, 0, nsteps_f);

      {
        int adp_native = 0; double adp_sum = 0; int adp_emu = 0;
        for(auto& r : adp_hist){
          if(r.native_fp64) adp_native++;
          else { adp_sum += r.slices; adp_emu++; }
        }
        double adp_avg = adp_emu > 0 ? adp_sum / adp_emu : 0.0;
        write_row("adp_dynamic", ta, ea, adp_avg, adp_native, (int)adp_hist.size());
      }

      {
        int ca_native = 0; double ca_sum = 0; int ca_emu = 0;
        for(int sl : adapt_hist){
          if(sl == 0) ca_native++;
          else { ca_sum += sl; ca_emu++; }
        }
        double ca_avg = ca_emu > 0 ? ca_sum / ca_emu : 0.0;
        write_row("custom_adaptive", tadapt, eadapt, ca_avg, ca_native, (int)adapt_hist.size());
      }

      fflush(csv_fp);
    }

    if (do_eq && d_scale_perm) CUDA_CHECK(cudaFree(d_scale_perm));
  }

  printf("\n================================================================\n");

  if(csv_fp){
    fclose(csv_fp);
    printf("  Results written to: %s\n", csv_path);
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_Ao)); CUDA_CHECK(cudaFree(d_Ap));
  CUDA_CHECK(cudaFree(d_Ar)); CUDA_CHECK(cudaFree(d_pw)); CUDA_CHECK(cudaFree(d_info));
  CUDA_CHECK(cudaFree(dor)); CUDA_CHECK(cudaFree(du)); CUDA_CHECK(cudaFree(dos));
  CUDA_CHECK(cudaFree(d_scale)); CUDA_CHECK(cudaFree(d_max_bits));
  CUDA_CHECK(cudaFree(d_adp_bits));
  CUDA_CHECK(cudaFree(d_fp_workspace));

  CUBLAS_CHECK(cublasDestroy(cb_native));
  CUBLAS_CHECK(cublasDestroy(cb_fixed));
  CUBLAS_CHECK(cublasDestroy(cb_adp));
  CUBLAS_CHECK(cublasDestroy(cb_adap));
  CUSOLVER_CHECK(cusolverDnDestroy(cs));
  CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
  CUDA_CHECK(cudaEventDestroy(analysis_done));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaStreamDestroy(analysis_stream));
  return 0;
}
