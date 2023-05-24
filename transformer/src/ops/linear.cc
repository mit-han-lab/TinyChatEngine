#include <cassert>

#include "operators.h"

template <typename T>
void linear(Matrix3D<T> &a, Matrix3D<T> &b, Matrix3D<T> &c) {
    // a: m x k   b: n x k   c: m x n
    assert(a.m_dim_x == b.m_dim_x);  // batch dim
    assert(a.m_dim_z == b.m_dim_z);  // k
    assert(a.m_dim_y == c.m_dim_y);  // m
    assert(b.m_dim_y == c.m_dim_z);  // n

    int m = a.m_dim_y, n = b.m_dim_y, k = a.m_dim_z, b_size = b.m_dim_x;

    for (int b_ = 0; b_ < b_size; b_++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                T acc = 0;
                for (int kk = 0; kk < k; k++) {
                    acc += a(b_, i, kk) * b(b_, j, kk);
                }

                c(b_, i, j) = acc;
            }
        }
    }
}

void Linear_FP::forward(const Matrix3D<float> &a, Matrix3D<float> &c) {
    Matrix3D<float> b = this->weight;
    const int m = a.m_dim_y, n = b.m_dim_y, k = a.m_dim_z, b_size = b.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START_FLOPS(profile_name, ops);

    // a: m x k   b: n x k   c: m x n
    assert(a.m_dim_x == b.m_dim_x);  // batch dim
    assert(a.m_dim_z == b.m_dim_z);  // k
    assert(a.m_dim_y == c.m_dim_y);  // m
    assert(b.m_dim_y == c.m_dim_z);  // n
    // batch dim == 1 only support MM for now
    assert(a.m_dim_x == 1);
    assert(b.m_dim_x == 1);

    struct matmul_params params;
    params.A.row = a.m_dim_y;
    params.A.column = a.m_dim_z;
    params.A.data_ptr = a.m_data;
    params.B.row = b.m_dim_y;
    params.B.column = b.m_dim_z;
    params.B.data_ptr = b.m_data;
    params.C.row = c.m_dim_y;
    params.C.column = c.m_dim_z;
    params.C.data_ptr = c.m_data;
    params.opt_params.blk_size = BLK_SIZE;
    params.opt_params.num_thread = NUM_THREAD;

    matmul::MatmulOperator op = matmul::MatmulOperator();
    op.mat_mul_transposed_fastover_column((const struct matmul_params *)&params);

    PROFILE_END(profile_name);
    return;
}

// void linear_fp(Matrix3D<float> &a, Matrix3D<float> &b, Matrix3D<float> &c) {
//     // a: m x k   b: n x k   c: m x n
//     assert(a.m_dim_x == b.m_dim_x);  // batch dim
//     assert(a.m_dim_z == b.m_dim_z);  // k
//     assert(a.m_dim_y == c.m_dim_y);  // m
//     assert(b.m_dim_y == c.m_dim_z);  // n
//     // batch dim == 1 only support MM for now
//     assert(a.m_dim_x == 1);
//     assert(b.m_dim_x == 1);

//     int m = a.m_dim_y, n = b.m_dim_y, k = a.m_dim_z, b_size = b.m_dim_x;

//     struct matmul_params params;
//     params.A.row = a.m_dim_y;
//     params.A.column = a.m_dim_z;
//     params.A.data_ptr = a.m_data;
//     params.B.row = b.m_dim_y;
//     params.B.column = b.m_dim_z;
//     params.B.data_ptr = b.m_data;
//     params.C.row = c.m_dim_y;
//     params.C.column = c.m_dim_z;
//     params.C.data_ptr = c.m_data;
//     params.opt_params.blk_size = BLK_SIZE;
//     params.opt_params.num_thread = NUM_THREAD;

//     matmul::MatmulOperator ops = matmul::MatmulOperator();
//     ops.mat_mul_fast((const struct matmul_params *)&params);

//     return;
// }
