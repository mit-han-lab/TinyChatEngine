#include <cmath>
#include <iomanip>

#include "operators.h"

// TODO: modify metal for weights
void LlamaRMSNorm_metal::forward(const Matrix3D<half> &x, Matrix3D<half> &output, float eps) {
    int m = x.m_dim_x * x.m_dim_y;
    int n = x.m_dim_z;
    dim3 grid(m);
    dim3 block(min(n, 1024));

    /* For general cases, n is equal to hidden_units, e.g., 512/1024.
       Since we have warp shuffle inside the code, block.x % 32 should be 0.
    */
    if (n % 32 != 0) {
        block.x = 1024;
    }

    block.x = block.x / (4 / sizeof(half));  // if using half, only need half of block.x

    /* should pay attention to the rsqrt precision */
    half *input = x.m_data, *out = output.m_data;
    float *gamma = weight.m_data;

    matmul::MatmulOperator op = matmul::MatmulOperator();
    op.rms_norm_metal(input, gamma, out, eps, m, n);  // For gpt-3
}