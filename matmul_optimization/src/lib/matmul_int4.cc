#include <cstdlib>
#include <iostream>

#include "matmul.h"

namespace matmul {

void MatmulOperator::naive_mat_mul_int4(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float B_zp = B->qparams.zero_point_float;
    float B_offset = B->qparams.offset;
    float B_sc = B->qparams.scale;
    float *data_A = A->data_ptr, *data_C = C->data_ptr;
    int8_t *data_B = B->int4_data_ptr;
    CHECK_MATRICES(A, B, C);

    for (i = 0; i < C->row; i++)
        for (j = 0; j < C->column; j++) {
            float acc = 0;
            for (k = 0; k < A->column; k++)
                acc += data_A[i * A->column + k] * (((float)data_B[k * B->column + j] - B_zp) * B_sc + B_offset);

            data_C[i * C->column + j] = acc;
        }
}
}  // namespace matmul
