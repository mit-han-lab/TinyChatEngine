#include <cstdlib>
#include <iostream>

#include "matmul.h"

namespace matmul {

void MatmulOperator::naive_mat_mul_int8(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int32_t A_zp = A->qparams.zero_point, C_zp = C->qparams.zero_point;
    float A_sc = A->qparams.scale, B_sc = B->qparams.scale, C_sc = C->qparams.scale;
    float effective_scale = A_sc * B_sc / C_sc;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr, *data_C = C->int8_data_ptr;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    CHECK_MATRICES(A, B, C);

    for (i = 0; i < C->row; i++)
        for (j = 0; j < C->column; j++) {
            int acc = 0;
            for (k = 0; k < A->column; k++)
                acc += ((int32_t)data_A[i * A->column + k] - A_zp) * data_B[k * B->column + j];

            acc = (int32_t)((float)acc * effective_scale);
            acc -= C_zp;
            acc = MAX(acc, q_min);
            acc = MIN(acc, q_max);
            data_C[i * C->column + j] = (int8_t)acc;
        }
}
}  // namespace matmul
