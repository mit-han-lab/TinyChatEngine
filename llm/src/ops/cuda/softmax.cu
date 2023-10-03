#include <cmath>
#include "operators.h"

__global__ void softmax_cuda(Matrix3D<half> input, Matrix3D<half> output) {
    // Calculate indices i, j in the input array
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < input.m_dim_x && j < input.m_dim_y) {
        // half max_value = __float2half(-INFINITY);
        half max_value = -65504;
        half sum = 0;

        // Find the maximum value in the input array
        for (int k = 0; k < input.m_dim_z; k++) {
            half value = input(i, j, k);
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ >= 860  // CCompute Capability >= 8.6
            max_value = __hmax(max_value, value);
#else
            max_value = __hgt(max_value, value) ? max_value : value;
#endif
#endif
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
