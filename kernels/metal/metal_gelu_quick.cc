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
void MatmulOperator::gelu_quick_metal(const struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *C = &params->C;

    MetalMatMulParams matmulparams = {m: (unsigned int)A->row, n: (unsigned int)C->column, k: (unsigned int)A->column};
    MetalMatmulBuffers bufferparams = {A: A->data_ptr, C: C->data_ptr};
    MetalIMP::run_gelu_quick(matmulparams, &bufferparams);
};
}  // namespace matmul