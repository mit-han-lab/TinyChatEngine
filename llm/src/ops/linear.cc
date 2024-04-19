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

#ifdef USE_ACCELERATE
#define MAX_WEIGHT_BUFFER 32000 * 4096
static float *w_fp32;
void Linear_FP_int4::initialize_weight_memory() {
    allocate_aligned_memory(w_fp32, MAX_WEIGHT_BUFFER * sizeof(float));
}
#endif

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
    params.B.row = b.m_dim_z;     // k
    params.B.column = b.m_dim_y;  // n
    params.B.data_ptr = b.m_data;
    params.C.row = c.m_dim_y;
    params.C.column = c.m_dim_z;
    params.C.data_ptr = c.m_data;
    params.opt_params.blk_size = BLK_SIZE;
    params.opt_params.num_thread = NUM_THREAD;

    matmul::MatmulOperator op = matmul::MatmulOperator();
#ifndef QM_CUDA  // not support yet
    if (this->has_bias) {
        params.bias.row = this->bias.m_dim_y;
        params.bias.column = this->bias.m_dim_z;
        params.bias.data_ptr = this->bias.m_data;
        op.mat_mul_accelerator_transposed_fastover_column_bias((const struct matmul_params *)&params);
    } else
#endif
        op.mat_mul_accelerator_transposed_fastover_column((const struct matmul_params *)&params);

    PROFILE_END(profile_name);
    return;
}

void Linear_FP_int4::forward_ref(const Matrix3D<float> &a, Matrix3D<float> &c) {
    Matrix3D<uint8_t> b = this->weight;
    const int m = a.m_dim_y, n = b.m_dim_y, k = a.m_dim_z, b_size = b.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START_FLOPS(profile_name, ops);

    // a: m x k   b: n x k   c: m x n
    assert(a.m_dim_x == b.m_dim_x);      // batch dim
    assert(a.m_dim_z / 2 == b.m_dim_z);  // k
    assert(a.m_dim_y == c.m_dim_y);      // m
    assert(b.m_dim_y == c.m_dim_z);      // n
    // batch dim == 1 only support MM for now
    assert(a.m_dim_x == 1);
    assert(b.m_dim_x == 1);

    struct matmul_params params;
    params.A.row = a.m_dim_y;
    params.A.column = a.m_dim_z;
    params.A.data_ptr = a.m_data;
    params.B.row = b.m_dim_y;
    params.B.column = b.m_dim_z;
    params.B.int4_data_ptr = b.m_data;
    params.C.row = c.m_dim_y;
    params.C.column = c.m_dim_z;
    params.C.data_ptr = c.m_data;
    params.opt_params.blk_size = BLK_SIZE;
    params.opt_params.num_thread = NUM_THREAD;
    params.scales = this->scale.m_data;
    params.offset = this->offset.m_data;
    params.zero_point = this->zero_point.m_data;
    params.block_size = QK;

    matmul::MatmulOperator op = matmul::MatmulOperator();
    op.naive_mat_mul_int4((const struct matmul_params *)&params);

    PROFILE_END(profile_name);
    return;
}

void Linear_FP_int4::forward_fast(const Matrix3D<float> &x, Matrix3D<float> &output) {
    const int num_thread = NUM_THREAD;
    Matrix3D<uint8_t> b = this->weight;
    const int m = x.m_dim_y, n = b.m_dim_y, k = x.m_dim_z, b_size = b.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START_FLOPS(profile_name, ops);

    // a: m x k   b: n x k   c: m x n
    assert(output.m_dim_x == 1);
    assert(output.m_dim_y == x.m_dim_y);
    assert(output.m_dim_z == weight.m_dim_y);
    assert(x.m_dim_z / 2 == weight.m_dim_z);

    assert(output.m_dim_z > num_thread);
    assert(output.m_dim_z % (num_thread * 2) == 0);  // unroll column by 2

    struct matmul_params params;
    params.A.row = x.m_dim_y;
    params.A.column = x.m_dim_z;
    params.A.data_ptr = x.m_data;
    params.B.row = b.m_dim_z;     // k
    params.B.column = b.m_dim_y;  // n
    params.B.int4_data_ptr = b.m_data;
    params.C.row = output.m_dim_y;
    params.C.column = output.m_dim_z;
    params.C.data_ptr = output.m_data;
    params.opt_params.num_thread = num_thread;
    params.scales = this->scale.m_data;
    params.offset = this->offset.m_data;
    params.block_size = QK;

    matmul::MatmulOperator op = matmul::MatmulOperator();
    op.mat_mul_accelerator_int4_fast(&params);

    PROFILE_END(profile_name);
    return;
}

#ifdef USE_INT8_INT4_PRODUCT
static int8_t *x_int8;
static float *x_scale;
void Linear_FP_int4::initialize_memory(const int block_size) {
#ifdef PACK_QK
    allocate_aligned_memory(x_int8,
                            (MAX_LINEAR_LENGTH) * sizeof(int8_t) + (MAX_LINEAR_LENGTH / block_size) * sizeof(float));
#else
    allocate_aligned_memory(x_int8, MAX_LINEAR_LENGTH * sizeof(int8_t));
    allocate_aligned_memory(x_scale, (MAX_LINEAR_LENGTH / block_size) * sizeof(float));
#endif  // PACK_QK
}
#endif  // USE_INT8_INT4_PRODUCT

void Linear_FP_int4::forward(const Matrix3D<float> &x, Matrix3D<float> &output) {
    const int num_thread = NUM_THREAD;
    Matrix3D<uint8_t> b = this->weight;
    const int m = x.m_dim_y, n = b.m_dim_y, k = x.m_dim_z, b_size = b.m_dim_x;
    const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START_FLOPS(profile_name, ops);

    // a: m x k   b: n x k   c: m x n
    assert(output.m_dim_x == 1);
    assert(output.m_dim_y == x.m_dim_y);
    assert(output.m_dim_z == weight.m_dim_y);
    assert(x.m_dim_z / 2 == weight.m_dim_z);

    assert(output.m_dim_z > num_thread);
    // assert(output.m_dim_z % (num_thread * 2) == 0);  // unroll column by 2

    struct matmul_params params;
    params.A.row = x.m_dim_y;
    params.A.column = x.m_dim_z;
    params.A.data_ptr = x.m_data;
    params.B.row = b.m_dim_z;     // k
    params.B.column = b.m_dim_y;  // n
    params.B.int4_data_ptr = b.m_data;
    params.C.row = output.m_dim_y;
    params.C.column = output.m_dim_z;
    params.C.data_ptr = output.m_data;
    params.opt_params.num_thread = num_thread;
    params.scales = this->scale.m_data;
    params.offset = this->offset.m_data;
    params.block_size = QK;

    if (this->has_bias) params.bias.data_ptr = this->bias.m_data;

    matmul::MatmulOperator op = matmul::MatmulOperator();
#ifdef USE_INT8_INT4_PRODUCT
    if (!x_int8) this->initialize_memory(params.block_size);
    params.A.int8_data_ptr = x_int8;
    params.A_scales = x_scale;
#ifdef PACK_QK
    params.B.int4_data_ptr = (uint8_t *)this->packed_weights;
#endif
#ifndef QM_CUDA  // not support yet
    if (!this->has_bias)
        params.bias.data_ptr = NULL;
    else
        params.bias.data_ptr = this->bias.m_data;
#endif
#ifdef USE_ACCELERATE
    if (!w_fp32) this->initialize_weight_memory();
    params.B.data_ptr = w_fp32;
    if (params.A.row <= 100) {
        op.mat_mul_accelerator_int8_int4_fast_no_offset(&params);
    } else {
        params.alpha = 1.0;
        op.cblas_gemm_accelerator_no_offset(&params);
    }
#else
    op.mat_mul_accelerator_int8_int4_fast_no_offset(&params);
#endif
#else
    op.mat_mul_accelerator_int4_fast_no_offset(&params);
#endif

    PROFILE_END(profile_name);
    return;
}
