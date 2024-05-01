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
void MatmulOperator::batch_add_metal(const struct matmul_params *params, unsigned int m_dim_x, unsigned int m_dim_y, unsigned int m_dim_z) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    MetalMatMulParams matmulparams = {m_dim_x: m_dim_x, m_dim_y: m_dim_y, m_dim_z: m_dim_z};
    MetalMatmulBuffers bufferparams = {A: A->data_ptr, C: C->data_ptr, B: (unsigned char*)B->data_ptr};
    MetalIMP::run_batch_add(matmulparams, &bufferparams);
};
}