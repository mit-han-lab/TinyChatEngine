#include "operators.cuh"

__global__ void batch_Add_half(Matrix3D<float> input, Matrix3D<float> input2, Matrix3D<float> output) {
    // Find the maximum value in the input array
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Ensure we are not out of bounds
    if (i < input.m_dim_x && j < input.m_dim_y && k < input.m_dim_z) {
        output(i, j, k) = input(i, j, k) + input2(0, j, k);
    }
}
