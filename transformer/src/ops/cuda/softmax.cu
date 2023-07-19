#include <cmath>
#include "operators.cuh"

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    
    return __int_as_float(old);
}

// __global__ void softmax_half(Matrix3D<float> input, Matrix3D<float> output) {
//     // Calculate indices i, j in the input array
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i < input.m_dim_x && j < input.m_dim_y) {
//         // // Shared memory for the maximum value and sum
//         // __shared__ float max_value;
//         // __shared__ float sum;

//         // if (threadIdx.x == 0 && threadIdx.y == 0) {
//         //     max_value = input(0, 0, 0);
//         //     sum = 0.0f;
//         // }
//         // __syncthreads();

//         float max_value = input.m_data[0];
//         float sum = 0.0f;
//         __syncthreads();

//         // Find the maximum value in the input array
//         for (int k = 0; k < input.m_dim_z; k++) {
//             float value = input(i, j, k);
//             atomicMax(&max_value, value);
//         }
//         __syncthreads();

//         // Compute the sum
//         for (int k = 0; k < input.m_dim_z; k++) {
//             float value = expf(input(i, j, k) - max_value);
//             atomicAdd(&sum, value);
//         }
//         __syncthreads();

//         // Compute the final softmax values
//         for (int k = 0; k < input.m_dim_z; k++) {
//             output(i, j, k) = expf(input(i, j, k) - max_value) / sum;
//         }
//     }
// }


__global__ void softmax_half(Matrix3D<float> input, Matrix3D<float> output) {
    // Calculate indices i, j in the input array
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < input.m_dim_x && j < input.m_dim_y) {
        float max_value = -INFINITY;
        float sum = 0;

        // Find the maximum value in the input array
        for (int k = 0; k < input.m_dim_z; k++) {
            float value = input(i, j, k);
            // atomicMax(&max_value, value);
            max_value = fmaxf(max_value, value);
        }
        // __syncthreads();

        // Compute the sum
        for (int k = 0; k < input.m_dim_z; k++) {
            float value = input(i, j, k);
            // atomicAdd(&sum, value);
            sum += __expf(value - max_value);
        }
        // __syncthreads();

        // Compute the final softmax values
        for (int k = 0; k < input.m_dim_z; k++) {
            float value = input(i, j, k);
            output(i, j, k) = __expf(value - max_value) / sum;
        }
    }
}
