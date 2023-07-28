#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>

#include <iostream>
#include <string>

#include "matmul.h"

namespace matmul {

void MatmulOperator::CHECK_MATRICES(const struct matrix *A, const struct matrix *B, const struct matrix *C) {
    assert(A->column == B->row);
    assert(C->column == B->column);
    assert(C->row == A->row);
}

void MatmulOperator::CHECK_MATRICES_int4weight(const struct matrix *A, const struct matrix *B, const struct matrix *C) {
    assert(B->row * B->column == A->column * C->column / 2);
    assert(C->row == A->row);
}

void MatmulOperator::mat_mul_transposed(const struct matmul_params *params) {
    int i, j, k;

    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

    for (i = 0; i < C->row; i++)
        for (j = 0; j < C->column; j++) {
            float acc = 0;
            for (k = 0; k < A->column; k++) acc += data_A[i * A->column + k] * data_B[j * B->column + k];
            data_C[i * C->column + j] = acc;
        }
}

float interval_to_ms(struct timeval *start, struct timeval *end) {
    float us_seconds = (end->tv_sec - start->tv_sec) * 1000000 + (end->tv_usec - start->tv_usec);
    return us_seconds / 1000;
}

}  // namespace matmul
