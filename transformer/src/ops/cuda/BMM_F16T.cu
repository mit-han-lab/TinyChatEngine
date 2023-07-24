#include "operators.cuh"
#include "utils.h"
#include "utils.cuh"
#include "lib/matmul.h"

__global__ void mat_mul_transposed_cuda_float(Matrix3D<float> A, Matrix3D<float> B, Matrix3D<float> C, const float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int m = A.m_dim_y, n = B.m_dim_y, k = A.m_dim_z;

    for (int bz = 0; bz < A.m_dim_x; bz++) {
        if (i < C.m_dim_y && j < C.m_dim_z) {
            float* data_A = A.m_data, *data_B = B.m_data, *data_C = C.m_data;

            float acc = 0;
            for (int k = 0; k < A.m_dim_z; k++) {
                acc += data_A[i * A.m_dim_z + k] * data_B[j * B.m_dim_z + k];
            }
            
            data_C[i * C.m_dim_z + j] = alpha * acc;
        }

        A.m_data += m * k;
        B.m_data += k * n;
        C.m_data += m * n;
    }
}

__global__ void mat_mul_transposed_cuda(Matrix3D<half> A, Matrix3D<half> B, Matrix3D<half> C, const half alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int m = A.m_dim_y, n = B.m_dim_y, k = A.m_dim_z;

    for (int bz = 0; bz < A.m_dim_x; bz++) {
        if (i < C.m_dim_y && j < C.m_dim_z) {
            half* data_A = A.m_data, *data_B = B.m_data, *data_C = C.m_data;

            half acc = __float2half(0.0f);
            for (int k = 0; k < A.m_dim_z; k++) {
                acc = __hfma(data_A[i * A.m_dim_z + k], data_B[j * B.m_dim_z + k], acc);
            }
            
            data_C[i * C.m_dim_z + j] = __hmul(alpha, acc);
        }

        A.m_data += m * k;
        B.m_data += k * n;
        C.m_data += m * n;
    }
}

void load_BMM_F16T(BMM_F16T &op, std::string prefix) { read_to_array((prefix + "/alpha.bin").c_str(), &op.alpha, 1); }

BMM_F16T::BMM_F16T(float _alpha) { this->alpha = _alpha; }

void BMM_F16T::forward(const Matrix3D<half> &a, const Matrix3D<half> &weight, Matrix3D<half> &c) {
    const Matrix3D<half> b = weight;
    // const int m = a.m_dim_y, n = b.m_dim_y, k = a.m_dim_z, b_size = b.m_dim_x;
    // const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
    PROFILE_START(profile_name);

    // a: m x k   b: n x k   c: m x n
    assert(a.m_dim_x == b.m_dim_x);  // batch dim
    assert(a.m_dim_z == b.m_dim_z);  // k
    assert(a.m_dim_y == c.m_dim_y);  // m
    assert(b.m_dim_y == c.m_dim_z);  // n

    struct matmul_params params;
    params.A.row = a.m_dim_y;
    params.A.column = a.m_dim_z;
    params.A.half_data_ptr = a.m_data;
    params.B.row = b.m_dim_y;
    params.B.column = b.m_dim_z;
    params.B.half_data_ptr = b.m_data;
    params.C.row = c.m_dim_y;
    params.C.column = c.m_dim_z;
    params.C.half_data_ptr = c.m_data;
    params.half_alpha = __float2half(alpha);

    dim3 block(16, 16);
    dim3 grid((params.C.row + block.x - 1) / block.x, (params.C.column + block.y - 1) / block.y);
    mat_mul_transposed_cuda<<<grid, block>>>(a, weight, c, params.half_alpha);

    PROFILE_END(profile_name);
}
