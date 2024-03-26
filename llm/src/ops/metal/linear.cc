#include <cassert>
#include "operators.h"
#include "utils.h"


// TODO: incorporate gemv from llama.cpp
void Linear_half_int4::forward(const Matrix3D<float16_t> &x, Matrix3D<float16_t> &output) {
    const int num_thread = 8;
    Matrix3D<int> b = this->weight;
    PROFILE_START(profile_name);

    // a: m x k   b: n x k   c: m x n
    assert(output.m_dim_x == 1);
    assert(output.m_dim_y == x.m_dim_y);
    // assert(output.m_dim_z == weight.m_dim_y);
    // assert(x.m_dim_z / 8 == weight.m_dim_z);

    assert(output.m_dim_z > num_thread);
    assert(output.m_dim_z % (num_thread * 2) == 0);  // unroll column by 2

    struct matmul_params params;
    params.A.row = x.m_dim_y;
    params.A.column = x.m_dim_z;
    params.A.half_data_ptr = x.m_data;
    params.B.row = b.m_dim_z;     // k
    params.B.column = b.m_dim_y;  // n
    params.B.int32_data_ptr = b.m_data;
    params.C.row = output.m_dim_y;
    params.C.column = output.m_dim_z;
    params.C.half_data_ptr = output.m_data;
    params.opt_params.num_thread = num_thread;
    params.half_scales = this->scale.m_data;
    params.int32_zero_point = this->zero_point.m_data;
    params.block_size = QK;

    matmul::MatmulOperator op = matmul::MatmulOperator();
    op.mat_mul_int4_f32_metal(&params); //BUG: gemv and matmul int4? (llama.cpp matmul needed)

    PROFILE_END(profile_name);
    return;
}