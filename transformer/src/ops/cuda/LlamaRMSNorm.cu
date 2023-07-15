#include <cmath>
#include <iomanip>

#include "operators.cuh"
#include "utils.h"

__global__ void LlamaRMSNorm_half_kernel(const Matrix3D_cuda<float> &x, const Matrix3D_cuda<float>& weight, Matrix3D_cuda<float> &output, float eps) {
    int i = blockIdx.x; // batches
    int j = threadIdx.x; // samples

    if (i < x.m_dim_x && j < x.m_dim_y) {
        float var = 0;

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

void LlamaRMSNorm_half::forward(const Matrix3D_cuda<float> &x, Matrix3D_cuda<float> &output) {
    dim3 numBlocks(x.m_dim_x, 1, 1);
    dim3 numThreads(x.m_dim_y, 1, 1);

    LlamaRMSNorm_half_kernel<<<numBlocks, numThreads>>>(x, weight, output, eps);
    cudaDeviceSynchronize();
}
