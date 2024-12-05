#include <cstdlib>
#include <iostream>

#include "matmul_cuda.h"

namespace matmul {

void MatmulOperatorCUDA::naive_mat_mul_fp16_int4(const struct matmul_params *params) {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    // CHECK_MATRICES_int4weight(A, B, C);

    naive_float16_t weight;
    for (int i = 0; i < C->row; i++) {
        for (int j = 0; j < C->column; j++) {
            naive_float16_t acc = (naive_float16_t)0.0;

            for (int k = 0; k < B->row; k++) {
                naive_float16_t s = params->fp16_scales[(k / block_size) * C->column + j];
                naive_float16_t z = static_cast<naive_float16_t>(8.0f); // TODO: support dynamic zeropoint
                naive_float16_t input = A->fp16_data_ptr[i * A->column + k];

                // order of weights is 0 2 4 6 1 3 5 7
                if (j % 8 == 0)
                    weight = ((naive_float16_t)(B->int32_data_ptr[k * B->column + (j / 8)] & 0x0000000F) - z) * s;
                else if (j % 8 == 1)
                    weight = ((naive_float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0x000F0000) >> 16) - z) * s;
                else if (j % 8 == 2)
                    weight = ((naive_float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0x000000F0) >> 4) - z) * s;
                else if (j % 8 == 3)
                    weight = ((naive_float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0x00F00000) >> 20) - z) * s;
                else if (j % 8 == 4)
                    weight = ((naive_float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0x00000F00) >> 8) - z) * s;
                else if (j % 8 == 5)
                    weight = ((naive_float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0x0F000000) >> 24) - z) * s;
                else if (j % 8 == 6)
                    weight = ((naive_float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0x0000F000) >> 12) - z) * s;
                else if (j % 8 == 7)
                    weight = ((naive_float16_t)((B->int32_data_ptr[k * B->column + (j / 8)] & 0xF0000000) >> 28) - z) * s;

                acc += input * weight;
                // printf("naive_mat_mul_fp16_int4 - s: %f, input: %f, weight: %f, acc: %f\n", static_cast<float>(s), static_cast<float>(input), static_cast<float>(weight), static_cast<float>(acc));
            }

            C->fp16_data_ptr[i * C->column + j] = acc;
        }
    }
}

}  // namespace matmul
