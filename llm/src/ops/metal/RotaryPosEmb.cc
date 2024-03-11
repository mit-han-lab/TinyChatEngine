#include <cmath>
#include "operators.h"

// TODO: match constants on metal
void RotaryPosEmb_cuda_forward(Matrix3D<half> query, Matrix3D<half> key, Matrix3D<half> cos, Matrix3D<half> sin, int start_idx, int len) {
    struct matmul_params params;
    params.A.row = query.m_dim_y;
    params.A.column = query.m_dim_z;
    params.A.half_data_ptr = query.m_data;
    params.B.row = key.m_dim_z;     // k
    params.B.column = key.m_dim_y;  // n
    params.B.int32_data_ptr = key.m_data;
    params.C.row = output.m_dim_y;
    params.C.column = output.m_dim_z;
    params.C.half_data_ptr = output.m_data;
    params.opt_params.num_thread = num_thread;
    params.half_scales = this->scale.m_data;
    params.int32_zero_point = this->zero_point.m_data;
    params.block_size = QK;

    matmul::MatmulOperator op = matmul::MatmulOperator();
    op.rope_metal(&params, query.m_dim_x, query.m_dim_y, query.m_dim_z, n_past, n_dims, mode, n_orig_ctx, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow ); 
    // In llama.cpp: 
    // 
}
