#include <cmath>
#include "operators.cuh"
// #include "operators.h"

// __device__ static float atomicMax(float* address, float val)
// {
//     int* address_as_i = (int*) address;
//     int old = *address_as_i, assumed;
//     do {
//         assumed = old;
//         old = ::atomicCAS(address_as_i, assumed,
//             __float_as_int(::fmaxf(val, __int_as_float(assumed))));
//     } while (assumed != old);
    
//     return __int_as_float(old);
// }

// __global__ void softmax_float(Matrix3D<float> input, Matrix3D<float> output) {
//     // Calculate indices i, j in the input array
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;

//     if (i < input.m_dim_x && j < input.m_dim_y) {
//         float max_value = -INFINITY;
//         float sum = 0;

//         // Find the maximum value in the input array
//         for (int k = 0; k < input.m_dim_z; k++) {
//             float value = input(i, j, k);
//             // atomicMax(&max_value, value);
//             max_value = fmaxf(max_value, value);
//         }
//         // __syncthreads();

//         // Compute the sum
//         for (int k = 0; k < input.m_dim_z; k++) {
//             float value = input(i, j, k);
//             // atomicAdd(&sum, value);
//             sum += __expf(value - max_value);
//         }
//         // __syncthreads();

//         // Compute the final softmax values
//         for (int k = 0; k < input.m_dim_z; k++) {
//             float value = input(i, j, k);
//             output(i, j, k) = __expf(value - max_value) / sum;
//         }
//     }
// }

__global__ void softmax_cuda(Matrix3D<half> input, Matrix3D<half> output) {
    // Calculate indices i, j in the input array
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < input.m_dim_x && j < input.m_dim_y) {
        // half max_value = __float2half(-INFINITY);
        half max_value = __float2half(-65504.0f);
        half sum = __float2half(0.0f);

        // Find the maximum value in the input array
        for (int k = 0; k < input.m_dim_z; k++) {
            half value = input(i, j, k);
            max_value = __hmax(max_value, value);
        }

        // Compute the sum
        for (int k = 0; k < input.m_dim_z; k++) {
            half value = input(i, j, k);
            // atomicAdd(&sum, value);
            sum = __hadd(sum, hexp(__hsub(value, max_value)));
            // sum = __hfma(__hsub(value, max_value), sum, sum);  // TODO: Check if this is correct and faster
        }

        // Compute the final softmax values
        for (int k = 0; k < input.m_dim_z; k++) {
            half value = input(i, j, k);
            output(i, j, k) = __hdiv(hexp(__hsub(value, max_value)), sum);
        }
    }
}
