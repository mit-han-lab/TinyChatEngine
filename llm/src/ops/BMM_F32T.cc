#include "operators.h"
#include "utils.h"

void load_BMM_F32T(BMM_F32T &op, std::string prefix) { read_to_array((prefix + "/alpha.bin").c_str(), &op.alpha, 1); }

BMM_F32T::BMM_F32T(float _alpha) { this->alpha = _alpha; }

void BMM_F32T::forward(const Matrix3D<float> &a, const Matrix3D<float> &weight, Matrix3D<float> &c) {
    const Matrix3D<float> b = weight;
    const int m = a.m_dim_y, n = b.m_dim_y, k = a.m_dim_z, b_size = b.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START_FLOPS(profile_name, ops);

    // a: m x k   b: n x k   c: m x n
    assert(a.m_dim_x == b.m_dim_x);  // batch dim
    assert(a.m_dim_z == b.m_dim_z);  // k
    assert(a.m_dim_y == c.m_dim_y);  // m
    assert(b.m_dim_y == c.m_dim_z);  // n

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
    params.alpha = alpha;

    matmul::MatmulOperator op = matmul::MatmulOperator();

    for (int bz = 0; bz < a.m_dim_x; bz++) {
        // if (params.A.column % 8 == 0) // TODO: debug this
        //     op.mat_mul_transposed_fastover_column((const struct matmul_params
        //     *)&params);
        // else
        op.mat_mul_transposed(&params);  // TODO: optimize this
        // TODO: apply SIMD here
        for (int i = 0; i < m * n; i++) {
            params.C.data_ptr[i] *= this->alpha;
        }
        params.A.data_ptr += m * k;
        params.B.data_ptr += k * n;
        params.C.data_ptr += m * n;
    }

    PROFILE_END(profile_name);
}

void BMM_F32T::forward_weight_untransposed(const Matrix3D<float> &a, const Matrix3D<float> &weight,
                                           Matrix3D<float> &c) {
    const Matrix3D<float> b = weight;
    const int m = a.m_dim_y, n = c.m_dim_z, k = a.m_dim_z, b_size = b.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START_FLOPS(profile_name, ops);

    // a: m x k   b: n x k   c: m x n
    assert(a.m_dim_x == b.m_dim_x);  // batch dim
    assert(a.m_dim_z == b.m_dim_y);  // k
    assert(a.m_dim_y == c.m_dim_y);  // m
    assert(b.m_dim_z == c.m_dim_z);  // n

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
    params.alpha = alpha;

    matmul::MatmulOperator op = matmul::MatmulOperator();

    for (int i = 0; i < m * n * a.m_dim_x; i++) {
        params.C.data_ptr[i] = 0;
    }

    for (int bz = 0; bz < a.m_dim_x; bz++) {
        float *data_A = params.A.data_ptr + bz * m * k, *data_B = params.B.data_ptr + bz * k * n,
              *data_C = params.C.data_ptr + bz * m * n;
        for (int i = 0; i < m; i++)
            for (int kk = 0; kk < k; kk++) {
                float Aikk0 = data_A[i * k + kk];
                for (int j = 0; j < n; j++) {
                    float Bjk0 = data_B[kk * n + j];
                    data_C[i * n + j] += Aikk0 * Bjk0;
                }
            }
    }

    PROFILE_END(profile_name);
}
