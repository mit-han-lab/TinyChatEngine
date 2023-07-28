#include <stdint.h>
#include <sys/time.h>

#include "half.hpp"  // Third-party header
typedef half_float::half naive_float16_t;

////// TODO: Fix this
#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_fp16.h>
    #include <cuda_runtime.h>
    typedef half float16_t;
#elif defined(__ARM_NEON)
    typedef __fp16 float16_t;
#elif defined(__x86_64__)
    printf("x86_64 does not natively support fp16, so we use `half_float` library to support fp16 through software-based solution.\n");
    typedef half_float::half float16_t;
#else
    printf("Unsupported platform (we only support CUDA, Arm, and x86_64). Using uint16_t as float16_t.\n");
    typedef uint16_t float16_t;
#endif

// TODO: deprecate this
#define MAX_TRANSPOSE_BUFFER 2048 * 2048
#define RUNS 1
static float transpose_tmp[MAX_TRANSPOSE_BUFFER];

// Data structures
struct quantization_params {
    float scale;
    bool per_channel = false;
    int32_t zero_point;
    int8_t q_min = -128, q_max = 127;
};

struct matrix {
    int row;
    int column;
    float *data_ptr;
    float16_t *half_data_ptr;
    naive_float16_t *fp16_data_ptr;
    int *int32_data_ptr;
    int8_t *int8_data_ptr;
    uint8_t *uint8_data_ptr;
    uint8_t *int4_data_ptr;
    struct quantization_params qparams;
    int length() { return row * column; }
};

struct optimization_params {
    int blk_size;
    int num_thread = 8;
};

struct matmul_params {
    struct matrix A, B, C, bias;
    struct optimization_params opt_params;
    float alpha, beta;
    float16_t half_alpha;
    // for int4
    float *scales, *offset, *zero_point;
    float16_t *half_scales;
    naive_float16_t *fp16_scales;
    int *int32_zero_point;
    int block_size;
};

struct thread_args {
    const struct matrix *A;
    const struct matrix *B;
    const struct matrix *C;
    const struct matmul_params *params;
    int start_i, end_i, blk_size;
};

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))
namespace matmul {
class MatmulOperator {
   public:
    void naive_mat_mul(const struct matmul_params *params);
    void mat_mul_unrolling(const struct matmul_params *params);
    void mat_mul_reordering(const struct matmul_params *params);
    void mat_mul_tiling(const struct matmul_params *params);
    void mat_mul_multithreading(const struct matmul_params *params);
    void mat_mul_transpose(const struct matmul_params *params);
    void mat_mul_transposed(const struct matmul_params *params);
    void mat_mul_accelerator_transposed_fastover_column(const struct matmul_params *params);
    void mat_mul_accelerator_transpose_simd(const struct matmul_params *params);
    void mat_mul_accelerator_fast(const struct matmul_params *params);
    void mat_mul_onednn(const struct matmul_params *params);
    void mat_mul_onednn_int8(const struct matmul_params *params);
    // int8
    void naive_mat_mul_int8(const struct matmul_params *params);
    void mat_mul_accelerator_int8(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_32unroll_over_column(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32_batch(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32_over_column(const struct matmul_params *params);
    void mat_mul_accelerator_int8_fast_2x2_omp(const struct matmul_params *params);
    // int4
    void mat_mul_accelerator_int4_fast(const struct matmul_params *params);
    void mat_mul_accelerator_int4_fast_no_offset(const struct matmul_params *params);
    void naive_mat_mul_int4(const struct matmul_params *params);
    void naive_mat_mul_int4_with_offset(const struct matmul_params *params);
    void naive_mat_mul_fp16_int4(const struct matmul_params *params);
    // cuda
    void mat_mul_cuda(const struct matmul_params *params);
    void gemm_forward_cuda(const struct matmul_params *params, int split_k_iters);
    void gemm_forward_cuda_half(const struct matmul_params *params, int split_k_iters);
    void gemm_forward_cuda_half_test(const struct matmul_params *params, int split_k_iters);

   private:
    float interval_to_us(struct timeval *start, struct timeval *end);
    void CHECK_MATRICES(const struct matrix *A, const struct matrix *B, const struct matrix *C);
    void CHECK_MATRICES_int4weight(const struct matrix *A, const struct matrix *B, const struct matrix *C);
};
}  // namespace matmul
