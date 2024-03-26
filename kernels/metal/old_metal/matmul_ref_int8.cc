#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"

namespace matmul {
void int8_ref_matmul(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr, *data_C = C->int8_data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * k + kk] * data_B[j * k + kk];
            }
            acc = (int32_t)std::round((float)acc * alpha + (float)(params->bias.int8_data_ptr[j]) * beta);
            acc = MAX(acc, q_min);
            acc = MIN(acc, q_max);
            data_C[i * n + j] = (int8_t)acc;
        }
    }
}

void int8_ref_matmul_nobias(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr, *data_C = C->int8_data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * A->column + kk] * data_B[j * B->row + kk];
            }
            acc = (int32_t)std::round((float)acc * alpha);
            acc = MAX(acc, q_min);
            acc = MIN(acc, q_max);
            data_C[i * C->column + j] = (int8_t)acc;
        }
    }
}

void int8_ref_matmul_nobias_batch(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr, *data_C = C->int8_data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * A->column + kk] * data_B[i * B->row * B->column + j * B->row + kk];
            }
            acc = (int32_t)std::round((float)acc * alpha);
            acc = MAX(acc, q_min);
            acc = MIN(acc, q_max);
            data_C[i * C->column + j] = (int8_t)acc;
        }
    }
}

void int8_ref_matmul_bfp32_ofp32(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr;
    float *data_C = C->data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * A->column + kk] * data_B[j * B->row + kk];
            }
            data_C[i * C->column + j] = (float)acc * alpha + (float)(params->bias.data_ptr[j]);
        }
    }
}

void int8_ref_matmul_nobias_ofp32(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr;
    float *data_C = C->data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * A->column + kk] * data_B[j * B->row + kk];
            }
            data_C[i * C->column + j] = (float)acc * alpha;
        }
    }
}

void int8_ref_matmul_nobias_ofp32_batch(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr;
    float *data_C = C->data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    assert(A->column == B->row);
    assert(C->row == A->row);
    assert(C->column == B->column);
    int m = A->row, n = B->column, k = A->column;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int kk = 0; kk < k; kk++) {
                acc += data_A[i * A->column + kk] * data_B[i * B->row * B->column + j * B->row + kk];
            }
            data_C[i * C->column + j] = (float)acc * alpha;
        }
    }
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll(const struct matmul_params *params) {
    int8_ref_matmul(params);
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias(const struct matmul_params *params) {
    int8_ref_matmul_nobias(params);
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch(const struct matmul_params *params) {
    int8_ref_matmul_nobias_batch(params);
}

void MatmulOperator::mat_mul_accelerator_int8_fast_32unroll_over_column(const struct matmul_params *params) {
    int8_ref_matmul(params);
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32(const struct matmul_params *params) {
    int8_ref_matmul_bfp32_ofp32(params);
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32(const struct matmul_params *params) {
    int8_ref_matmul_nobias_ofp32(params);
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32_batch(const struct matmul_params *params) {
    int8_ref_matmul_nobias_ofp32_batch(params);
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32_over_column(
    const struct matmul_params *params) {
    int8_ref_matmul_bfp32_ofp32(params);
}

}  // namespace matmul
