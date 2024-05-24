#include "../../../include/operators.h"
#include "utils.h"
#include "metal_compute.h"

void load_BMM_F16T(BMM_F16T &op, std::string prefix) { read_to_array_half((prefix + "/alpha_half.bin").c_str(), &op.alpha, 1); }

BMM_F16T::BMM_F16T(half _alpha) { this->alpha = _alpha; }

void BMM_F16T::forward(const Matrix3D<half> &a, const Matrix3D<half> &weight, Matrix3D<half> &c) {
    const Matrix3D<half> b = weight;
    PROFILE_START(profile_name);

    // a: m x k   b: n x k   c: m x n
    assert(a.m_dim_x == b.m_dim_x);  // batch dim
    assert(a.m_dim_z == b.m_dim_z);  // k
    assert(a.m_dim_y == c.m_dim_y);  // m
    assert(b.m_dim_y == c.m_dim_z);  // n

    struct metal_params params;
    params.A.row = a.m_dim_y;
    params.A.column = a.m_dim_z;
    params.A.half_data_ptr = a.m_data;
    params.B.row = b.m_dim_y;
    params.B.column = b.m_dim_z;
    params.B.half_data_ptr = b.m_data;
    params.C.row = c.m_dim_y;
    params.C.column = c.m_dim_z;
    params.C.half_data_ptr = c.m_data;
    params.half_alpha = alpha;
    params.bs = a.m_dim_x;

    params.op = METAL_KERNEL_MUL_MM_INT4_F32;
    add_node(&params);
    PROFILE_END(profile_name);
}

void BMM_F16T::forward_weight_untransposed(const Matrix3D<half> &a, const Matrix3D<half> &weight, Matrix3D<half> &c) {
    const Matrix3D<half> b = weight;
    PROFILE_START(profile_name);

    // a: m x k   b: n x k   c: m x n
    assert(a.m_dim_x == b.m_dim_x);  // batch dim
    assert(a.m_dim_z == b.m_dim_y);  // k
    assert(a.m_dim_y == c.m_dim_y);  // m
    assert(b.m_dim_z == c.m_dim_z);  // n

    struct metal_params params;
    params.A.row = a.m_dim_y;
    params.A.column = a.m_dim_z;
    params.A.half_data_ptr = a.m_data;
    params.B.row = b.m_dim_y;
    params.B.column = b.m_dim_z;
    params.B.half_data_ptr = b.m_data;
    params.C.row = c.m_dim_y;
    params.C.column = c.m_dim_z;
    params.C.half_data_ptr = c.m_data;
    params.op = METAL_KERNEL_MUL_MM_INT4_F32;
    add_node(&params);
    PROFILE_END(profile_name);
}