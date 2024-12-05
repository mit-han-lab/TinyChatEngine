#include <cassert>
#include "operators.h"
#include "utils.h"

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
    // params.offset = this->offset.m_data;  // TODO: Currently, we don't need offset
    params.int32_zero_point = this->zero_point.m_data;
    params.block_size = QK;

    matmul::MatmulOperator &op = matmul::CreateMatmulOperator();
    op.gemv_forward_cuda(&params);

    PROFILE_END(profile_name);
    return;
}


void Linear_FP16_int4_ref::forward_ref(const Matrix3D<naive_float16_t> &a, Matrix3D<naive_float16_t> &c) {
    Matrix3D<int> b = this->weight;
    PROFILE_START(profile_name);

    // a: m x k   b: n x k   c: m x n
    assert(a.m_dim_x == b.m_dim_x);      // batch dim
    assert(a.m_dim_z == b.m_dim_z);      // k
    assert(a.m_dim_y == c.m_dim_y);      // m
    assert(b.m_dim_y == c.m_dim_z / 8);  // n

    // batch dim == 1 only support MM for now
    assert(a.m_dim_x == 1);
    assert(b.m_dim_x == 1);

    struct matmul_params params;
    params.A.row = a.m_dim_y;
    params.A.column = a.m_dim_z;
    params.A.fp16_data_ptr = a.m_data;
    params.B.row = b.m_dim_z;
    params.B.column = b.m_dim_y;
    params.B.int32_data_ptr = b.m_data;
    params.C.row = c.m_dim_y;
    params.C.column = c.m_dim_z;
    params.C.fp16_data_ptr = c.m_data;
    params.fp16_scales = this->scale.m_data;
    // params.offset = this->offset.m_data;   // TODO: Currently, we don't need offset
    params.int32_zero_point = this->zero_point.m_data;
    params.block_size = QK;

    matmul::MatmulOperator &op = matmul::CreateMatmulOperator();
    op.naive_mat_mul_fp16_int4((const struct matmul_params *)&params);

    PROFILE_END(profile_name);
    return;
}
