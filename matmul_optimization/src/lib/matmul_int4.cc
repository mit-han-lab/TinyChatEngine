#include <cstdlib>
#include <iostream>

#include "matmul.h"

namespace matmul {

void MatmulOperator::naive_mat_mul_fp16_int4(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    // float16_t *B_sc = params->fp16_scales;
    // int *B_zp = params->int32_zero_point;
    // float16_t *data_C = C->data_ptr;
    CHECK_MATRICES_int4weight(A, B, C);

    // std::cout << "naive_mat_mul_int4 -- A->row: " << A->row << " A->column: " << A->column 
    //           << "; B->row: " << B->row << " B->column: " << B->column 
    //           << "; C->row: " << C->row << " C->column: " << C->column << std::endl;

    float16_t weight;
    for (i = 0; i < C->row; i++) {
        for (j = 0; j < C->column; j++) {
            float16_t acc = (float16_t)0.0;

            for (int k = 0; k < B->row; k++) {
                float16_t s = params->fp16_scales[(k / block_size) * C->column + j];
                float16_t input = A->fp16_data_ptr[i * A->column + k];

                if (j % 8 == 0)
                    weight = ((float16_t)(B->int32_data_ptr[k * B->column + (j / 8)] & 0x0000000F) - 8.0) * s;
                else if (j % 8 == 1)
                    weight = ((float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0x000000F0) >> 4) - 8.0) * s;
                else if (j % 8 == 2)
                    weight = ((float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0x00000F00) >> 8) - 8.0) * s;
                else if (j % 8 == 3)
                    weight = ((float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0x0000F000) >> 12) - 8.0) * s;
                else if (j % 8 == 4)
                    weight = ((float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0x000F0000) >> 16) - 8.0) * s;
                else if (j % 8 == 5)
                    weight = ((float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0x00F00000) >> 20) - 8.0) * s;
                else if (j % 8 == 6)
                    weight = ((float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0x0F000000) >> 24) - 8.0) * s;
                else if (j % 8 == 7)
                    weight = ((float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0xF0000000) >> 28) - 8.0) * s;

                acc += input * weight;
                // printf("naive_mat_mul_fp16_int4 - s: %f, input: %f, weight: %f, acc: %f\n", static_cast<float>(s), static_cast<float>(input), static_cast<float>(weight), static_cast<float>(acc));
            }

            C->fp16_data_ptr[i * C->column + j] = acc;
        }
    }
}


void MatmulOperator::naive_mat_mul_int4(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *B_sc = params->scales;
    float *B_zp = params->zero_point;
    float *data_C = C->data_ptr;
    CHECK_MATRICES_int4weight(A, B, C);

    // std::cout << "naive_mat_mul_int4 -- A->row: " << A->row << " A->column: " << A->column 
    //           << "; B->row: " << B->row << " B->column: " << B->column 
    //           << "; C->row: " << C->row << " C->column: " << C->column << std::endl;

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

    // std::cout << "naive_mat_mul_int4_with_offset -- A->row: " << A->row << " A->column: " << A->column 
    //           << "; B->row: " << B->row << " B->column: " << B->column 
    //           << "; C->row: " << C->row << " C->column: " << C->column << std::endl;

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
