#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <cmath>
#include <cstdlib>
// #include <omp.h>
#include <arm_neon.h>

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

#include "common.h"
#include "../matmul.h"
#include "pthread_pool.h"

struct fp32_thread_args {
    int start_i, end_i, start_j, end_j, tile_size;
    const struct matmul_params* params;
};

namespace matmul {
void fp32_ref_matmul(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * k + kk] * data_B[j * k + kk];
            }
            acc = acc;
            data_C[i * n + j] = acc;
        }
    }
}

void fp32_ref_matmul_bias(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *bias = params->bias.data_ptr;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * k + kk] * data_B[j * k + kk];
            }
            acc = acc + bias[j];
            data_C[i * n + j] = acc;
        }
    }
}

#ifdef USE_ACCELERATE
inline void fp32_matmul_transposed_cblas_gemm(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
    float alpha = params->alpha;

    assert(A->column == B->column);
    assert(C->row == A->row);
    assert(C->column == B->row);
    int m = C->row, n = C->column, k = A->column;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k,
                alpha, data_A, k,
                       data_B, k,
                0.0f,  data_C, n);
}

inline void fp32_matmul_untransposed_cblas_gemm(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = C->row, n = C->column, k = A->column;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0f, data_A, k,
                      data_B, n,
                0.0f, data_C, n);
}

void fp32_matmul_bias_cblas_gemm(const struct matmul_params *params) {
    // struct fp32_thread_args* mat_args = (struct fp32_thread_args*)args;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *bias = params->bias.data_ptr;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k,
                1.0f,   data_A, k,
                        data_B, k,
                0.0f,   data_C, n);
                         
    for (int i = 0; i < m; i++) {
        vDSP_vadd(bias, 1, data_C + i * n, 1, data_C + i * n, 1, n);
    }
}
#endif

void MatmulOperator::mat_mul_accelerator_transposed_fastover_column(const struct matmul_params *params) {
#ifdef USE_ACCELERATE
    fp32_matmul_transposed_cblas_gemm(params);
#else
    fp32_ref_matmul(params);
#endif
}

void MatmulOperator::mat_mul_accelerator_untransposed_fastover_column(const struct matmul_params *params) {
#ifdef USE_ACCELERATE
    fp32_matmul_untransposed_cblas_gemm(params);
#endif
}

inline static void* fp32_matmul_bias_optimized_gemm(void* args) {
    struct fp32_thread_args* mat_args = (struct fp32_thread_args*)args;
    const struct matmul_params* params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *bias = params->bias.data_ptr;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = C->row, n = C->column, k = A->column;
    int start_i = mat_args->start_i, end_i = mat_args->end_i;

    int TILE_SIZE = mat_args->tile_size;
    // assert((end_i - start_i) % TILE_SIZE == 0);
    assert(A->column % TILE_SIZE == 0);
    assert(C->column % TILE_SIZE == 0);
    assert(TILE_SIZE % 4 == 0);

    for (int ti = start_i; ti < end_i - TILE_SIZE + 1; ti += TILE_SIZE) {
        for (int tj = 0; tj < n - TILE_SIZE + 1; tj += TILE_SIZE) {
            for (int i = ti; i < ti + TILE_SIZE; i++) {
                for (int j = tj; j < tj + TILE_SIZE; j+=4) {
                    // float acc0[4] = {}, acc1[4] = {}, acc2[4] = {}, acc3[4] = {};
                    // float32x4_t *acc0_fp_128 = (float32x4_t*)acc0;
                    // float32x4_t *acc1_fp_128 = (float32x4_t*)acc1;
                    // float32x4_t *acc2_fp_128 = (float32x4_t*)acc2;
                    // float32x4_t *acc3_fp_128 = (float32x4_t*)acc3;
                    // for (int kk = 0; kk < k; kk+=4) {
                    //     float32x4_t Aik_Aik3 = vld1q_f32(&data_A[i * k + kk]);
                    //     float32x4_t val;
                    //     val = vmulq_f32(Aik_Aik3, vld1q_f32(&data_B[j * k + kk]));
                    //     *acc0_fp_128 = vaddq_f32(*acc0_fp_128, val);

                    //     val = vmulq_f32(Aik_Aik3, vld1q_f32(&data_B[(j+1) * k + kk]));
                    //     *acc1_fp_128 = vaddq_f32(*acc1_fp_128, val);

                    //     val = vmulq_f32(Aik_Aik3, vld1q_f32(&data_B[(j+2) * k + kk]));
                    //     *acc2_fp_128 = vaddq_f32(*acc2_fp_128, val);

                    //     val = vmulq_f32(Aik_Aik3, vld1q_f32(&data_B[(j+3) * k + kk]));
                    //     *acc3_fp_128 = vaddq_f32(*acc3_fp_128, val);
                    // }
                    // data_C[i * n + j] = acc0[0] + acc0[1] + acc0[2] + acc0[3] + bias[j];
                    // data_C[i * n + j + 1] = acc1[0] + acc1[1] + acc1[2] + acc1[3] + bias[j + 1];
                    // data_C[i * n + j + 2] = acc2[0] + acc2[1] + acc2[2] + acc2[3] + bias[j + 2];
                    // data_C[i * n + j + 3] = acc3[0] + acc3[1] + acc3[2] + acc3[3] + bias[j + 3];

                    float32x4_t acc0_fp_128 = vdupq_n_f32(0.0f);
                    float32x4_t acc1_fp_128 = vdupq_n_f32(0.0f);
                    float32x4_t acc2_fp_128 = vdupq_n_f32(0.0f);
                    float32x4_t acc3_fp_128 = vdupq_n_f32(0.0f);
                    for (int kk = 0; kk < k; kk+=4) {
                        float32x4_t Aik_Aik3 = vld1q_f32(&data_A[i * k + kk]);
                        acc0_fp_128 = vmlaq_f32(acc0_fp_128, Aik_Aik3, vld1q_f32(&data_B[j * k + kk]));
                        acc1_fp_128 = vmlaq_f32(acc1_fp_128, Aik_Aik3, vld1q_f32(&data_B[(j+1) * k + kk]));
                        acc2_fp_128 = vmlaq_f32(acc2_fp_128, Aik_Aik3, vld1q_f32(&data_B[(j+2) * k + kk]));
                        acc3_fp_128 = vmlaq_f32(acc3_fp_128, Aik_Aik3, vld1q_f32(&data_B[(j+3) * k + kk]));
                    }
                    data_C[i * n + j] = vaddvq_f32(acc0_fp_128) + bias[j];
                    data_C[i * n + j + 1] = vaddvq_f32(acc1_fp_128) + bias[j + 1];
                    data_C[i * n + j + 2] = vaddvq_f32(acc2_fp_128) + bias[j + 2];
                    data_C[i * n + j + 3] = vaddvq_f32(acc3_fp_128) + bias[j + 3];
                }
            }
        }
    }

    // Leftover rows
    int left_start_i = (end_i / TILE_SIZE) * TILE_SIZE;
    // for (int i = left_start_i; i < end_i; i++) {
    //     for (int j = 0; j < n; j++) {
    //         float acc = 0;
    //         for (int kk = 0; kk < k; kk++) {
    //             acc += data_A[i * k + kk] * data_B[j * k + kk];
    //         }
    //         acc = acc + bias[j];
    //         data_C[i * n + j] = acc;
    //     }
    // }
    ///////
    // for (int i = left_start_i; i < end_i; i++) {
    //     for (int tj = 0; tj < n - TILE_SIZE + 1; tj += TILE_SIZE) {
    //         for (int j = tj; j < tj + TILE_SIZE; j+=4) {
    //             float32x4_t acc0_fp_128 = vdupq_n_f32(0.0f);
    //             float32x4_t acc1_fp_128 = vdupq_n_f32(0.0f);
    //             float32x4_t acc2_fp_128 = vdupq_n_f32(0.0f);
    //             float32x4_t acc3_fp_128 = vdupq_n_f32(0.0f);
    //             for (int kk = 0; kk < k; kk+=4) {
    //                 float32x4_t Aik_Aik3 = vld1q_f32(&data_A[i * k + kk]);
    //                 acc0_fp_128 = vmlaq_f32(acc0_fp_128, Aik_Aik3, vld1q_f32(&data_B[j * k + kk]));
    //                 acc1_fp_128 = vmlaq_f32(acc1_fp_128, Aik_Aik3, vld1q_f32(&data_B[(j+1) * k + kk]));
    //                 acc2_fp_128 = vmlaq_f32(acc2_fp_128, Aik_Aik3, vld1q_f32(&data_B[(j+2) * k + kk]));
    //                 acc3_fp_128 = vmlaq_f32(acc3_fp_128, Aik_Aik3, vld1q_f32(&data_B[(j+3) * k + kk]));
    //             }
    //             data_C[i * n + j] = vaddvq_f32(acc0_fp_128) + bias[j];
    //             data_C[i * n + j + 1] = vaddvq_f32(acc1_fp_128) + bias[j + 1];
    //             data_C[i * n + j + 2] = vaddvq_f32(acc2_fp_128) + bias[j + 2];
    //             data_C[i * n + j + 3] = vaddvq_f32(acc3_fp_128) + bias[j + 3];
    //         }
    //     }
    // }
    ///////
    for (int i = left_start_i; i < end_i; i++) {
        for (int j = 0; j < n; j+=4) {
            float32x4_t acc0_fp_128 = vdupq_n_f32(0.0f);
            float32x4_t acc1_fp_128 = vdupq_n_f32(0.0f);
            float32x4_t acc2_fp_128 = vdupq_n_f32(0.0f);
            float32x4_t acc3_fp_128 = vdupq_n_f32(0.0f);
            for (int kk = 0; kk < k; kk+=4) {
                float32x4_t Aik_Aik3 = vld1q_f32(&data_A[i * k + kk]);
                acc0_fp_128 = vmlaq_f32(acc0_fp_128, Aik_Aik3, vld1q_f32(&data_B[j * k + kk]));
                acc1_fp_128 = vmlaq_f32(acc1_fp_128, Aik_Aik3, vld1q_f32(&data_B[(j+1) * k + kk]));
                acc2_fp_128 = vmlaq_f32(acc2_fp_128, Aik_Aik3, vld1q_f32(&data_B[(j+2) * k + kk]));
                acc3_fp_128 = vmlaq_f32(acc3_fp_128, Aik_Aik3, vld1q_f32(&data_B[(j+3) * k + kk]));
            }
            data_C[i * n + j] = vaddvq_f32(acc0_fp_128) + bias[j];
            data_C[i * n + j + 1] = vaddvq_f32(acc1_fp_128) + bias[j + 1];
            data_C[i * n + j + 2] = vaddvq_f32(acc2_fp_128) + bias[j + 2];
            data_C[i * n + j + 3] = vaddvq_f32(acc3_fp_128) + bias[j + 3];
        }
    }

    return NULL;
}

void MatmulOperator::mat_mul_accelerator_transposed_fastover_column_bias(const struct matmul_params *params) {
#ifdef USE_ACCELERATE
    fp32_matmul_bias_cblas_gemm(params);
#else
    fp32_ref_matmul_bias(params);
#endif

    // int i, j, k;
    // const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    // assert(A->row == C->row);
    // const int num_thread = params->opt_params.num_thread;
    // struct fp32_thread_args threads_args[num_thread];
    // static void *pool = pool_start(fp32_matmul_bias_optimized_gemm, num_thread);
    // // Thread creation
    // for (j = 0; j < num_thread; j++) {
    //     threads_args[j].start_i = j * (params->C.row / num_thread);
    //     if (j == num_thread - 1) {
    //         threads_args[j].end_i = params->C.row;
    //     } else {
    //         threads_args[j].end_i = (j + 1) * (params->C.row / num_thread);
    //     }
    //     threads_args[j].tile_size = 32;
    //     threads_args[j].params = params;
    //     pool_enqueue(pool, &threads_args[j], '\0');
    // }
    // // Join threads
    // pool_wait(pool);
}

}  // namespace matmul
