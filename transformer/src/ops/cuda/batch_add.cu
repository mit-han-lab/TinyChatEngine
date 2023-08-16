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

    //// half version
    if (i < input.m_dim_x && j < input.m_dim_y && k < input.m_dim_z) {
        output(i, j, k) = __hadd(input(i, j, k), input2(0, j, k));
    }
}

__global__ void batch_Add_cuda_half2(Matrix3D<half> input, Matrix3D<half> input2, Matrix3D<half> output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < input.m_dim_x && j < input.m_dim_y && k < input.m_dim_z / 2) {
        half2* input_half2 = reinterpret_cast<half2*>(input.m_data);
        half2* input2_half2 = reinterpret_cast<half2*>(input2.m_data);
        half2* output_half2 = reinterpret_cast<half2*>(output.m_data);
        int input_half2_dim_y = input.m_dim_y;
        int input_half2_dim_z = input.m_dim_z / 2;
        // int input2_half2_dim_y = input2.m_dim_y;
        int input2_half2_dim_z = input2.m_dim_z / 2;
        int output_half2_dim_y = output.m_dim_y;
        int output_half2_dim_z = output.m_dim_z / 2;

        output_half2[i * output_half2_dim_y * output_half2_dim_z + j * output_half2_dim_z + k] = 
                __hadd2(input_half2[i * input_half2_dim_y * input_half2_dim_z + j * input_half2_dim_z + k], 
                        input2_half2[j * input2_half2_dim_z + k]);
    }
}