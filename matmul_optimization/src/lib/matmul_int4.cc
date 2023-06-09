#include <cstdlib>
#include <iostream>

#include "matmul.h"

namespace matmul {

void MatmulOperator::naive_mat_mul_int4(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *B_sc = params->scales;
    float *B_offset = params->offset;
    float *B_zp = params->zero_point;
    float *data_A = A->data_ptr, *data_C = C->data_ptr;
    uint8_t *data_B = B->int4_data_ptr;
    CHECK_MATRICES_int4weight(A, B, C);

    for (i = 0; i < C->row; i++) {
        for (j = 0; j < C->column; j++) {
            float acc = 0;
            float deq_b;
            for (k = 0; k < A->column; k++)
                if (j < 32) {
                    if ((j / 16) % 2 == 0) 
                        deq_b = (float)(data_B[k * B->column + j] & 0x0F);
                    else
                        deq_b = (float)(data_B[k * B->column + j - 16] >> 4);
                }
                else {
                    if ((j / 16) % 2 == 0) {
                        if (j % 2 == 0)
                            deq_b = (float)(data_B[k * B->column + j / 2] & 0x0F);
                        else
                            deq_b = (float)(data_B[k * B->column + j / 2 + 1] & 0x0F);
                    }
                    else {
                        if (j % 2 == 0)
                            deq_b = (float)(data_B[k * B->column + (j - 16) / 2] >> 4);
                        else
                            deq_b = (float)(data_B[k * B->column + (j - 16) / 2 + 1] >> 4);
                    }
                }
                acc += data_A[i * A->column + k] * ((deq_b - *B_zp) * B_sc[(k * B->column + j) / block_size] + B_offset[(k * B->column + j) / block_size]);

            data_C[i * C->column + j] = acc;
        }
    }

    // Test
    std::cout << "B_sc[0] = " << B_sc[0] << std::endl;
    std::cout << "B_sc[1] = " << B_sc[1] << std::endl;
    std::cout << "B_sc[2] = " << B_sc[2] << std::endl;
    std::cout << "B_sc[-3] = " << B_sc[B->row * B->column * 2 / 32 - 3] << std::endl;
    std::cout << "B_sc[-2] = " << B_sc[B->row * B->column * 2 / 32 - 2] << std::endl;
    std::cout << "B_sc[-1] = " << B_sc[B->row * B->column * 2 / 32 - 1] << std::endl;
    std::cout << "B_zp = " << B_zp[0] << std::endl;
    std::cout << "B->row: " << B->row << std::endl;
    std::cout << "B->column: " << B->column << std::endl;
    std::cout << "B_offset[0] = " << B_offset[0] << std::endl;
    std::cout << "B_offset[1] = " << B_offset[1] << std::endl;
    std::cout << "B_offset[2] = " << B_offset[2] << std::endl;
    std::cout << "B_offset[-3] = " << B_offset[B->row * B->column * 2 / 32 - 3] << std::endl;
    std::cout << "B_offset[-2] = " << B_offset[B->row * B->column * 2 / 32 - 2] << std::endl;
    std::cout << "B_offset[-1] = " << B_offset[B->row * B->column * 2 / 32 - 1] << std::endl;
    
}
}  // namespace matmul
