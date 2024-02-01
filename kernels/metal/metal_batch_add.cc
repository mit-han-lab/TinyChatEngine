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
void MatmulOperator::batch_add_metal(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

    MetalMatMulParams matmulparams = {(unsigned int)A->row, (unsigned int)C->column, (unsigned int)A->column};
    MetalMatmulBuffers bufferparams = {A: A->data_ptr, C: C->data_ptr, B: (unsigned char*)B->data_ptr};
    MetalIMP::run_batch_add(matmulparams, &bufferparams);
};
}  // namespace matmul