#include <cmath>
#include "../../../include/operators.h"
#include "utils.h"
#include "metal_compute.h"

void RotaryPosEmb_metal_forward(Matrix3D<half> query, Matrix3D<half> key, Matrix3D<half> cos, Matrix3D<half> sin, int start_idx, int len) {
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

    params.n_orig_ctx = 4096;
    params.n_past = 0;
    params.n_dims = 128;
    params.mode = 0;
    params.freq_base = 10000.0;
    params.freq_scale = 1.0;
    params.ext_factor = 0.0;
    params.attn_factor = 1.0;
    params.beta_fast = 32.0;
    params.beta_slow = 1.0;
    params.op = METAL_KERNEL_ROPE;
    add_node(&params);
    return;
}
