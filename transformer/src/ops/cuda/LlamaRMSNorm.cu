#include <cmath>
#include <iomanip>

#include "operators.cuh"
#include "utils.h"

__global__ void LlamaRMSNorm_half_kernel(const Matrix3D<float> x, const Matrix3D<float> weight, Matrix3D<float> output, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < x.m_dim_x && j < x.m_dim_y) {
        float var = 0.0f;

        for (int k = 0; k < x.m_dim_z; k++) {  // hidden states
            var += x(i, j, k) * x(i, j, k);
        }

        var /= static_cast<float>(x.m_dim_z);
        float variance = 1.0 / sqrtf(var + eps);

        for (int k = 0; k < x.m_dim_z; k++) {
            float value = static_cast<float>(x(i, j, k));
            float fp_out = (value * variance) * weight(0, 0, k);
            output(i, j, k) = fp_out;
        }
    }
}

void LlamaRMSNorm_half::forward(const Matrix3D<float> &x, Matrix3D<float> &output) {
    dim3 block(32, 32);
    dim3 grid((x.m_dim_x + block.x - 1) / block.x, (x.m_dim_y + block.y - 1) / block.y);

    LlamaRMSNorm_half_kernel<<<grid, block>>>(x, weight, output, eps);
    // cudaDeviceSynchronize();
}
