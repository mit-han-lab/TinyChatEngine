#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"

namespace matmul {
void MatmulOperator::mat_mul_accelerator_int4_fast(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    assert(params->block_size == 32);  // support block size 32 for now

    for (i = 0; i < C->row; i++) {
        for (j = 0; j < C->column; j++) {
            float acc = 0;
            for (k = 0; k < B->row * 2; k += block_size) {
                float s = scale[j * (B->row / 16) + k / 32];   // /16:B->column is packed 4bits
                float o = offset[j * (B->row / 16) + k / 32];  // /16:B->column is packed 4bits
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->row + k / 2];
                float *x_ptr = &A->data_ptr[i * A->column + k];
                for (int qi = 0; qi < block_size / 2; qi += 4) {
                    uint8_t packed_int4_0 = weight_32_int4[qi];
                    uint8_t packed_int4_1 = weight_32_int4[qi + 1];
                    uint8_t packed_int4_2 = weight_32_int4[qi + 2];
                    uint8_t packed_int4_3 = weight_32_int4[qi + 3];
                    float deq_0 = (float)((packed_int4_0 & 0x0F) - 8.0) * s + o;
                    float deq_1 = (float)((packed_int4_1 & 0x0F) - 8.0) * s + o;
                    float deq_2 = (float)((packed_int4_2 & 0x0F) - 8.0) * s + o;
                    float deq_3 = (float)((packed_int4_3 & 0x0F) - 8.0) * s + o;
                    float deq_4 = (float)((packed_int4_0 >> 4) - 8.0) * s + o;
                    float deq_5 = (float)((packed_int4_1 >> 4) - 8.0) * s + o;
                    float deq_6 = (float)((packed_int4_2 >> 4) - 8.0) * s + o;
                    float deq_7 = (float)((packed_int4_3 >> 4) - 8.0) * s + o;
                    acc += *x_ptr++ * deq_0;
                    acc += *x_ptr++ * deq_1;
                    acc += *x_ptr++ * deq_2;
                    acc += *x_ptr++ * deq_3;
                    acc += *x_ptr++ * deq_4;
                    acc += *x_ptr++ * deq_5;
                    acc += *x_ptr++ * deq_6;
                    acc += *x_ptr++ * deq_7;
                }
            }
            C->data_ptr[i * C->column + j] = acc;
        }
    }
};

void MatmulOperator::mat_mul_accelerator_int4_fast_no_offset(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    assert(params->block_size == 32);  // support block size 32 for now

    for (i = 0; i < C->row; i++) {
        for (j = 0; j < C->column; j++) {
            float acc = 0;
            for (k = 0; k < B->row * 2; k += block_size) {
                float s = scale[j * (B->row / 16) + k / 32];  // /16:B->column is packed 4bits
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->row + k / 2];
                float *x_ptr = &A->data_ptr[i * A->column + k];
                for (int qi = 0; qi < block_size / 2; qi += 4) {
                    uint8_t packed_int4_0 = weight_32_int4[qi];
                    uint8_t packed_int4_1 = weight_32_int4[qi + 1];
                    uint8_t packed_int4_2 = weight_32_int4[qi + 2];
                    uint8_t packed_int4_3 = weight_32_int4[qi + 3];
                    float deq_0 = (float)((packed_int4_0 & 0x0F) - 8.0) * s;
                    float deq_1 = (float)((packed_int4_1 & 0x0F) - 8.0) * s;
                    float deq_2 = (float)((packed_int4_2 & 0x0F) - 8.0) * s;
                    float deq_3 = (float)((packed_int4_3 & 0x0F) - 8.0) * s;
                    float deq_4 = (float)((packed_int4_0 >> 4) - 8.0) * s;
                    float deq_5 = (float)((packed_int4_1 >> 4) - 8.0) * s;
                    float deq_6 = (float)((packed_int4_2 >> 4) - 8.0) * s;
                    float deq_7 = (float)((packed_int4_3 >> 4) - 8.0) * s;
                    acc += *x_ptr++ * deq_0;
                    acc += *x_ptr++ * deq_1;
                    acc += *x_ptr++ * deq_2;
                    acc += *x_ptr++ * deq_3;
                    acc += *x_ptr++ * deq_4;
                    acc += *x_ptr++ * deq_5;
                    acc += *x_ptr++ * deq_6;
                    acc += *x_ptr++ * deq_7;
                }
            }
            C->data_ptr[i * C->column + j] = acc;
        }
    }
};
}  // namespace matmul
