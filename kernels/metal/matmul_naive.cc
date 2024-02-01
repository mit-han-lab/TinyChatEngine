#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "matmul_metal_imp.h"

namespace matmul {
    // naive float*float matmul
void MatmulOperator::mat_mul_metal(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    assert(params->block_size % 32 == 0);  // support block size to be multiply of 32
    assert(A->row == C->row);              // support block size to be multiply of 32

    MetalMatMulParams matmulparams = {(unsigned int)A->row, (unsigned int)C->column, (unsigned int)A->column,
                                      (unsigned int)block_size};
    MetalMatmulBuffers bufferparams = {A->data_ptr, C->data_ptr, scale, offset, (unsigned char*)B->data_ptr};
    MetalIMP::run_mat_mul_accelerator_int4_fast_no_offset(matmulparams, &bufferparams);
};
}  // namespace matmul
