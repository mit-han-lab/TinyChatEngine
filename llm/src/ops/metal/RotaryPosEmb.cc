#include <cmath>
#include "../../../include/operators.h"
#include "utils.h"
#include "metal_compute.h"

void RotaryPosEmb_metal_forward(Matrix3D<half> query, Matrix3D<half> key, Matrix3D<half> cos, Matrix3D<half> sin, int start_idx, int len) {
    
    struct metal_params params_query;
    struct metal_params params_key;
    params_query.A.row = query.m_dim_y;
    params_query.A.column = query.m_dim_z;
    params_query.A.half_data_ptr = query.m_data;
    params_query.B.row = key.m_dim_z;     // k
    params_query.B.column = key.m_dim_y;  // n
    params_query.B.half_data_ptr = key.m_data;
    params_query.C.row = query.m_dim_y;
    params_query.C.column = query.m_dim_z;
    params_query.C.half_data_ptr = query.m_data;

    params_query.n_orig_ctx = 4096;
    params_query.n_past = 0;
    params_query.n_dims = 128;
    params_query.mode = 0;
    params_query.freq_base = 10000.0;
    params_query.freq_scale = 1.0;
    params_query.ext_factor = 0.0;
    params_query.attn_factor = 1.0;
    params_query.beta_fast = 32.0;
    params_query.beta_slow = 1.0;
    params_query.op = METAL_KERNEL_ROPE;
    add_node(&params_query);

    // TO DO: src1: indices
    params_key.A.row = key.m_dim_y;
    params_key.A.column = key.m_dim_z;
    params_key.A.half_data_ptr = key.m_data;
    params_key.B.row = key.m_dim_z;     // k
    params_key.B.column = key.m_dim_y;  // n
    params_key.B.half_data_ptr = key.m_data;
    params_key.C.row = key.m_dim_y;
    params_key.C.column = key.m_dim_z;
    params_key.C.half_data_ptr = key.m_data;

    params_key.n_orig_ctx = 4096;
    params_key.n_past = 0;
    params_key.n_dims = 128;
    params_key.mode = 0;
    params_key.freq_base = 10000.0;
    params_key.freq_scale = 1.0;
    params_key.ext_factor = 0.0;
    params_key.attn_factor = 1.0;
    params_key.beta_fast = 32.0;
    params_key.beta_slow = 1.0;
    params_key.op = METAL_KERNEL_ROPE;
    add_node(&params_query);
    return;
}
