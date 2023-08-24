#include <cmath>

#include "operators.h"
#include "utils.h"

void load_BMM_S8T_S8N_S8T(BMM_S8T_S8N_S8T &op, std::string prefix) {
    read_to_array((prefix + "/alpha.bin").c_str(), &op.alpha, 1);
}

BMM_S8T_S8N_S8T::BMM_S8T_S8N_S8T(struct BMM_S8T_S8N_S8T_params &op_params) { alpha = op_params.alpha; }

void BMM_S8T_S8N_S8T::forward(const Matrix3D<int8_t> &x, const Matrix3D<int8_t> &weight, Matrix3D<int8_t> &output) {
    const int m = x.m_dim_y, k = x.m_dim_z, n = weight.m_dim_y, b = x.m_dim_x;
    const long long ops = (long long)b * 2 * (long long)m * (long long)n * (long long)k + (long long)m * (long long)n;
    PROFILE_START_FLOPS(profile_name, ops);
    assert(output.m_dim_x == x.m_dim_x);
    assert(output.m_dim_y == x.m_dim_y);
    assert(output.m_dim_z == weight.m_dim_y);
    assert(x.m_dim_z == weight.m_dim_z);

    struct matmul_params params;

    params.A.row = m;
    params.A.column = k;
    params.A.int8_data_ptr = x.m_data;
    params.A.qparams.scale = alpha;  // effective_scale = a * B / C
    params.B.qparams.scale = 1.0;
    params.C.qparams.scale = 1.0;
    params.A.qparams.zero_point = 0;
    params.B.row = k;
    params.B.column = n;
    params.B.int8_data_ptr = weight.m_data;
    params.B.qparams.zero_point = 0;
    params.C.row = m;
    params.C.column = n;
    params.C.int8_data_ptr = output.m_data;
    params.C.qparams.zero_point = 0;
    params.opt_params.blk_size = BLK_SIZE;
    params.opt_params.num_thread = NUM_THREAD;
    params.C.qparams.q_max = 127;
    params.C.qparams.q_min = -128;
    params.alpha = alpha;

    matmul::MatmulOperator matmul_op = matmul::MatmulOperator();

    // process each batch
    if (m == 1 && x.m_dim_x > 1) {
        // merge each batch
        params.A.row = x.m_dim_x;
        params.C.row = x.m_dim_x;
        // B is batched, need a new op for this!
        matmul_op.mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch(&params);
    } else {
        for (int bz = 0; bz < x.m_dim_x; bz++) {
            matmul_op.mat_mul_accelerator_int8_fast_2x2_32unroll_nobias(&params);
            params.A.int8_data_ptr += m * k;
            params.B.int8_data_ptr += k * n;
            params.C.int8_data_ptr += m * n;
        }
    }

    PROFILE_END(profile_name);
}
