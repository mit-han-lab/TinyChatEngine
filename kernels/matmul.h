#ifndef MATMUL_H
#define MATMUL_H
#include <stdint.h>
#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#else
#include <sys/time.h>
#endif

#include "half.hpp"  // Third-party header
typedef half_float::half naive_float16_t;

#ifdef QM_CUDA
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
typedef half float16_t;
#elif defined(__ARM_NEON)
typedef __fp16 float16_t;
#elif defined(__x86_64__)
// x86_64 does not natively support fp16, so we use `half_float` library to support fp16 through software-based
// solution.
typedef half_float::half float16_t;
#else
// Unsupported platform (we only support CUDA, Arm, and x86_64). Using uint16_t as float16_t.
typedef uint16_t float16_t;
#endif

#ifdef QM_ARM
#ifdef __ARM_FEATURE_DOTPROD
#include <arm_neon.h>
// Native implementation using vdotq_s32 when available
static inline int32x4_t my_vdotq_s32(int32x4_t accum, int8x16_t a, int8x16_t b) { return vdotq_s32(accum, a, b); }

#else
#include <arm_neon.h>
// Fallback implementation when vdotq_s32 is not available
static inline int32x4_t my_vdotq_s32(int32x4_t accum, int8x16_t a, int8x16_t b) {
    // Multiply and widen results to 16-bit integers
    int16x8_t result_low = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    int16x8_t result_high = vmull_s8(vget_high_s8(a), vget_high_s8(b));

    // Sum pairs of 16-bit values and accumulate into 32-bit integers
    return vaddq_s32(accum, vaddq_s32(vaddl_s16(vget_low_s16(result_low), vget_high_s16(result_low)),
                                      vaddl_s16(vget_low_s16(result_high), vget_high_s16(result_high))));
}
#endif
#endif

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
    int32_t *int32_data_ptr;
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
    // for int8 activation
    float *A_scales;
    int8_t A_zero_point;
};

struct thread_args {
    const struct matrix *A;
    const struct matrix *B;
    const struct matrix *C;
    const struct matmul_params *params;
    int start_i, end_i, blk_size;
};

#ifndef MAX
#define MAX(A, B) ((A) > (B) ? (A) : (B))
#endif
#ifndef MIN
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif

namespace matmul {
class MatmulOperator {
   public:
    virtual ~MatmulOperator() = default;

    // Virtual methods for various matrix multiplication operations
    virtual void mat_mul_transposed(const struct matmul_params* params);
    virtual void mat_mul_accelerator_transposed_fastover_column(const struct matmul_params* params) {}
    virtual void mat_mul_accelerator_transposed_fastover_column_bias(const struct matmul_params* params) {}
    virtual void mat_mul_accelerator_untransposed_fastover_column(const struct matmul_params* params) {}

    // int8 operations
    virtual void naive_mat_mul_int8(const struct matmul_params* params);
    virtual void mat_mul_accelerator_int8_fast_32unroll_over_column(const struct matmul_params* params) {}
    virtual void mat_mul_accelerator_int8_fast_2x2_32unroll(const struct matmul_params* params) {}
    virtual void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias(const struct matmul_params* params) {}
    virtual void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch(const struct matmul_params* params) {}
    virtual void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32(const struct matmul_params* params) {}
    virtual void mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32_batch(const struct matmul_params* params) {}
    virtual void mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32(const struct matmul_params* params) {}
    virtual void mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32_over_column(const struct matmul_params* params) {}

    // int4 operations
    virtual void mat_mul_accelerator_int4_fast(const struct matmul_params* params) {}
    virtual void mat_mul_accelerator_int4_fast_no_offset(const struct matmul_params* params) {}
    virtual void mat_mul_accelerator_int8_int4_fast_no_offset(struct matmul_params* params) {}
    virtual void gemv_accelerator_int8_int4_fast_no_offset(struct matmul_params* params) {}
    virtual void gemm_accelerator_int8_int4_fast_no_offset(struct matmul_params* params) {}
    virtual void gemm_accelerator_int8_int4_fast_no_offset_v2(struct matmul_params* params) {}
    virtual void cblas_gemm_accelerator_no_offset(struct matmul_params* params) {}
    virtual void naive_mat_mul_int4(const struct matmul_params* params);
    virtual void naive_mat_mul_int4_with_offset(const struct matmul_params* params);

    // CUDA-specific operations
    virtual void naive_mat_mul_fp16_int4(const struct matmul_params* params) {}
    virtual void mat_mul_cuda(const struct matmul_params* params) {}
    virtual void gemm_forward_cuda(const struct matmul_params* params, int split_k_iters) {}
    virtual void gemm_forward_cuda_8splits(const struct matmul_params* params, float16_t* split_8_buffer) {}
    virtual void gemm_forward_cuda_half(const struct matmul_params* params, int split_k_iters) {}
    virtual void gemm_forward_cuda_half_test(const struct matmul_params* params, int split_k_iters) {}
    virtual void gemv_forward_cuda(const struct matmul_params* params) {}

   protected:
    // Protected constructor to prevent direct instantiation
    // Directly creating an object of this class is not allowed because of the empty constructor
    MatmulOperator() = default;

   private:
    // Delete copy constructor and assignment operator for safety
    MatmulOperator& operator=(const MatmulOperator&) = delete;

    float interval_to_us(struct timeval *start, struct timeval *end);
    void CHECK_MATRICES(const struct matrix *A, const struct matrix *B, const struct matrix *C);
    void CHECK_MATRICES_int4weight(const struct matrix *A, const struct matrix *B, const struct matrix *C);
};

MatmulOperator& CreateMatmulOperator();

}  // namespace matmul

#endif
