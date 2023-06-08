#include <cmath>

#include "operators.h"
#include "utils.h"

void load_W8A8B8O8LinearReLU_params(W8A8B8O8LinearReLU &op, std::string prefix) {
    read_to_array((prefix + "/weight.bin").c_str(), op.params.B.int8_data_ptr, op.params.B.length());
    read_to_array((prefix + "/bias_int8.bin").c_str(), op.params.bias.int8_data_ptr, op.params.bias.length());
    read_to_array((prefix + "/alpha.bin").c_str(), &op.alpha, 1);
    read_to_array((prefix + "/alpha.bin").c_str(), &op.params.alpha, 1);
    read_to_array((prefix + "/beta.bin").c_str(), &op.beta, 1);
    read_to_array((prefix + "/beta.bin").c_str(), &op.params.beta, 1);
}

W8A8B8O8LinearReLU::W8A8B8O8LinearReLU(struct W8A8B8O8LinearReLU_params &op_params) {
    Matrix3D<int8_t> weight = op_params.weight;
    Matrix3D<int8_t> bias = op_params.bias_int8;

    int k = weight.m_dim_z, n = weight.m_dim_y;
    params.A.qparams.scale = alpha;  // effective_scale = a * B / C
    params.B.qparams.scale = 1.0;
    params.C.qparams.scale = 1.0;
    params.A.qparams.zero_point = 0;
    params.B.row = k;
    params.B.column = n;
    params.B.int8_data_ptr = weight.m_data;
    params.B.qparams.zero_point = 0;
    params.C.qparams.zero_point = 0;
    params.opt_params.blk_size = BLK_SIZE;
    params.opt_params.num_thread = NUM_THREAD;
    params.C.qparams.q_max = 127;
    params.C.qparams.q_min = 0;
    params.bias.int8_data_ptr = bias.m_data;
    params.bias.row = 1;
    params.bias.column = bias.m_dim_z;
    params.alpha = alpha;
    params.beta = op_params.beta;
}

void W8A8B8O8LinearReLU::forward(const Matrix3D<int8_t> &x, Matrix3D<int8_t> &output) {
    const int m = x.m_dim_y, k = x.m_dim_z, n = params.B.column, b = x.m_dim_x;
    const long long ops = (long long)b * 2 * (long long)m * (long long)n * (long long)k + (long long)m * (long long)n;
    PROFILE_START_FLOPS(profile_name, ops);
    assert(output.m_dim_x == x.m_dim_x);
    assert(output.m_dim_y == x.m_dim_y);
    assert(output.m_dim_z == params.B.column);
    assert(x.m_dim_z == params.B.row);
    assert(output.m_dim_z == params.bias.column);

    params.A.row = m;
    params.A.column = k;
    params.A.int8_data_ptr = x.m_data;
    params.C.row = m;
    params.C.column = n;
    params.C.int8_data_ptr = output.m_data;
    params.A.qparams.scale = alpha;
    params.alpha = alpha;
    params.beta = beta;

    matmul::MatmulOperator matmul_op = matmul::MatmulOperator();

    if (m == 1) {
        // let's loop over the column dim instead of row
        for (int bz = 0; bz < x.m_dim_x; bz++) {
            matmul_op.mat_mul_accelerator_int8_fast_32unroll_over_column(&params);
            params.A.int8_data_ptr += m * k;
            params.C.int8_data_ptr += m * n;
        }
    } else {
        for (int bz = 0; bz < x.m_dim_x; bz++) {
            matmul_op.mat_mul_accelerator_int8_fast_2x2_32unroll(&params);
            params.A.int8_data_ptr += m * k;
            params.C.int8_data_ptr += m * n;
        }
    }

    PROFILE_END(profile_name);
}
