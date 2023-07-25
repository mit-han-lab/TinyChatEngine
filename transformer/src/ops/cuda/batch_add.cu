// #include "operators.cuh"
#include "operators.h"

// __global__ void batch_Add_float(Matrix3D<float> input, Matrix3D<float> input2, Matrix3D<float> output) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     int k = blockIdx.z * blockDim.z + threadIdx.z;

//     if (i < input.m_dim_x && j < input.m_dim_y && k < input.m_dim_z) {
//         output(i, j, k) = input(i, j, k) + input2(0, j, k);
//     }
// }

__global__ void batch_Add_cuda(Matrix3D<half> input, Matrix3D<half> input2, Matrix3D<half> output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < input.m_dim_x && j < input.m_dim_y && k < input.m_dim_z) {
        output(i, j, k) = __hadd(input(i, j, k), input2(0, j, k));
    }
}
