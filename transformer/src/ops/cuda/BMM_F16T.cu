#include "operators.cuh"
#include "utils.h"
#include "lib/matmul.h"

// __global__ void mat_mul_transposed_cuda(const struct matmul_params* params, const float alpha) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i < params->C.row && j < params->C.column) {
//         const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
//         float* data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

//         float acc = 0;
//         for (int k = 0; k < A->column; k++) {
//             acc += data_A[i * A->column + k] * data_B[j * B->column + k];
//         }
        
//         data_C[i * C->column + j] = alpha * acc;
//     }
// }

__global__ void mat_mul_transposed_cuda(Matrix3D<float> A, Matrix3D<float> B, Matrix3D<float> C, const float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    const int m = A.m_dim_y, n = B.m_dim_y, k = A.m_dim_z;

    // printf("1111111111\n");
    for (int bz = 0; bz < A.m_dim_x; bz++) {
        // printf("bz: %d\n", bz);
        if (i < C.m_dim_y && j < C.m_dim_z) {
            float* data_A = A.m_data, *data_B = B.m_data, *data_C = C.m_data;

            float acc = 0;
            for (int k = 0; k < A.m_dim_z; k++) {
                acc += data_A[i * A.m_dim_z + k] * data_B[j * B.m_dim_z + k];
            }
            
            data_C[i * C.m_dim_z + j] = alpha * acc;
        }

        // printf("2222222222\n");

        A.m_data += m * k;
        B.m_data += k * n;
        C.m_data += m * n;

        // printf("3333333333\n");
    }

    // printf("4444444444\n");
}

void load_BMM_F16T(BMM_F16T &op, std::string prefix) { read_to_array((prefix + "/alpha.bin").c_str(), &op.alpha, 1); }

BMM_F16T::BMM_F16T(float _alpha) { this->alpha = _alpha; }

void BMM_F16T::forward(const Matrix3D<float> &a, const Matrix3D<float> &weight, Matrix3D<float> &c) {
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
    // params.opt_params.blk_size = BLK_SIZE;
    // params.opt_params.num_thread = NUM_THREAD;
    params.alpha = alpha;

    matmul::MatmulOperator op = matmul::MatmulOperator();

    // for (int bz = 0; bz < a.m_dim_x; bz++) {
    //     // if (params.A.column % 8 == 0) // TODO: debug this
    //     //     op.mat_mul_transposed_fastover_column((const struct matmul_params
    //     //     *)&params);
    //     // else

    //     // op.mat_mul_transposed(&params);  // TODO: optimize this
    //     dim3 block(16, 16);  // You might want to optimize this size.
    //     dim3 grid((params.C.row + block.x - 1) / block.x, (params.C.column + block.y - 1) / block.y);
    //     // printf("bz: %d\n", bz);
    //     mat_mul_transposed_cuda<<<grid, block>>>(&params, this->alpha);
    //     cudaDeviceSynchronize();

    //     // TODO: apply SIMD here
    //     // for (int i = 0; i < m * n; i++) {
    //     //     params.C.data_ptr[i] *= this->alpha;
    //     // }
    //     params.A.data_ptr += m * k;
    //     params.B.data_ptr += k * n;
    //     params.C.data_ptr += m * n;
    // }

    dim3 block(16, 16);  // You might want to optimize this size.
    dim3 grid((params.C.row + block.x - 1) / block.x, (params.C.column + block.y - 1) / block.y);
    // printf("bz: %d\n", bz);
    // printf("000000000\n");
    mat_mul_transposed_cuda<<<grid, block>>>(a, weight, c, this->alpha);
    // cudaDeviceSynchronize();

    PROFILE_END(profile_name);
}


// __global__ void BMM_F16T_forward(const Matrix3D<float> a, const Matrix3D<float> weight, Matrix3D<float> c) {
//     const Matrix3D_cuda<float> b = weight;
//     const int m = a.m_dim_y, n = b.m_dim_y, k = a.m_dim_z, b_size = b.m_dim_x;
//     const long long ops = (long long)b_size * 2 * (long long)m * (long long)n * (long long)k;
//     PROFILE_START_FLOPS(profile_name, ops);

//     // a: m x k   b: n x k   c: m x n
//     assert(a.m_dim_x == b.m_dim_x);  // batch dim
//     assert(a.m_dim_z == b.m_dim_z);  // k
//     assert(a.m_dim_y == c.m_dim_y);  // m
//     assert(b.m_dim_y == c.m_dim_z);  // n

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
//     // params.opt_params.blk_size = BLK_SIZE;
//     // params.opt_params.num_thread = NUM_THREAD;
//     params.alpha = alpha;

//     matmul::MatmulOperator op = matmul::MatmulOperator();

//     for (int bz = 0; bz < a.m_dim_x; bz++) {
//         // if (params.A.column % 8 == 0) // TODO: debug this
//         //     op.mat_mul_transposed_fastover_column((const struct matmul_params
//         //     *)&params);
//         // else

//         // op.mat_mul_transposed(&params);  // TODO: optimize this
//         dim3 block(16, 16);  // You might want to optimize this size.
//         dim3 grid((params.C.row + block.x - 1) / block.x, (params.C.column + block.y - 1) / block.y);
//         // printf("bz: %d\n", bz);
//         mat_mul_transposed_cuda<<<grid, block>>>(&params, this->alpha);
//         cudaDeviceSynchronize();

//         // TODO: apply SIMD here
//         // for (int i = 0; i < m * n; i++) {
//         //     params.C.data_ptr[i] *= this->alpha;
//         // }
//         params.A.data_ptr += m * k;
//         params.B.data_ptr += k * n;
//         params.C.data_ptr += m * n;
//     }

//     PROFILE_END(profile_name);
// }
