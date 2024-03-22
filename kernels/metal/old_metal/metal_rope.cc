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
void MatmulOperator::rope_metal(const struct matmul_params *params, unsigned int m_dim_x, unsigned int m_dim_y, unsigned int m_dim_z,
int n_past, int n_dims, int mode, int n_orig_ctx, float freq_base, float freq_scale, float ext_factor, float attn_factor,
float beta_fast, float beta_slow) {
    int i, j, k;
    const struct matrix *A = &params->A, *C = &params->C;
    MetalMatMulParams matmulparams = {m_dim_x: m_dim_x, m_dim_y: m_dim_x, m_dim_z: m_dim_z, 
   n_past: n_past, n_dims: n_dims, mode: mode, n_orig_ctx: n_orig_ctx, freq_base: freq_base, freq_scale: freq_scale, ext_factor: ext_factor, 
   attn_factor: attn_factor, beta_fast: beta_fast, beta_slow: beta_slow, type_size: sizeof(short)}; //it uses half in cuda
    MetalMatmulBuffers bufferparams = {A: A->data_ptr, C: C->data_ptr};
    MetalIMP::run_rope(matmulparams, &bufferparams);
};
}