#include <cstdlib>
#include <iostream>

#include "matmul.h"

namespace matmul {
void MatmulOperator::naive_mat_mul_int4(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *B_sc = params->scales;
    float *B_zp = params->zero_point;
    float *data_C = C->data_ptr;
    CHECK_MATRICES_int4weight(A, B, C);

    for (i = 0; i < C->row; i++) {
        for (j = 0; j < C->column; j++) {
            float acc = 0;

            for (k = 0; k < A->column; k += block_size) {
                float s = B_sc[(j * B->column * 2 + k) / block_size];
                float z = *B_zp;
                float *data_A = &A->data_ptr[i * A->column + k];
                uint8_t *data_B = &B->int4_data_ptr[j * B->column + k / 2];

                for (int qi = 0; qi < block_size / 2; qi++) {
                    uint8_t packed_int4_0 = data_B[qi];
                    float deq_0 = ((float)(packed_int4_0 & 0x0F) - z) * s;
                    float deq_1 = ((float)(packed_int4_0 >> 4) - z) * s;
                    acc += *data_A++ * deq_0;
                    acc += *data_A++ * deq_1;
                }
            }

            data_C[i * C->column + j] = acc;
        }
    }
}

void MatmulOperator::naive_mat_mul_int4_with_offset(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *B_sc = params->scales;
    float *B_offset = params->offset;
    float *B_zp = params->zero_point;
    float *data_C = C->data_ptr;
    CHECK_MATRICES_int4weight(A, B, C);

    for (i = 0; i < C->row; i++) {
        for (j = 0; j < C->column; j++) {
            float acc = 0;

            for (k = 0; k < A->column; k += block_size) {
                float s = B_sc[(j * B->column * 2 + k) / block_size];
                float o = B_offset[(j * B->column * 2 + k) / block_size];
                float z = *B_zp;
                float *data_A = &A->data_ptr[i * A->column + k];
                uint8_t *data_B = &B->int4_data_ptr[j * B->column + k / 2];

                for (int qi = 0; qi < block_size / 2; qi++) {
                    uint8_t packed_int4_0 = data_B[qi];
                    float deq_0 = ((float)(packed_int4_0 & 0x0F) - z) * s + o;
                    float deq_1 = ((float)(packed_int4_0 >> 4) - z) * s + o;
                    acc += *data_A++ * deq_0;
                    acc += *data_A++ * deq_1;
                }
            }

            data_C[i * C->column + j] = acc;
        }
    }
}
}  // namespace matmul
