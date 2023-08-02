#include <cmath>
#include <iomanip>

#include "operators.h"
#include "utils.h"

// __global__ void LlamaRMSNorm_float_kernel(const Matrix3D<float> x, const Matrix3D<float> weight, Matrix3D<float> output, float eps) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
    
//     if (i < x.m_dim_x && j < x.m_dim_y) {
//         float var = 0.0f;

//         for (int k = 0; k < x.m_dim_z; k++) {  // hidden states
//             var += x(i, j, k) * x(i, j, k);
//         }

//         var /= static_cast<float>(x.m_dim_z);
//         float variance = 1.0 / sqrtf(var + eps);

//         for (int k = 0; k < x.m_dim_z; k++) {
//             float value = static_cast<float>(x(i, j, k));
//             float fp_out = (value * variance) * weight(0, 0, k);
            
//             output(i, j, k) = fp_out;
//         }
//     }
// }

__global__ void LlamaRMSNorm_cuda_kernel(const Matrix3D<half> x, const Matrix3D<float> weight, Matrix3D<half> output, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // half epf_half = __float2half(1e-5f);
    
    if (i < x.m_dim_x && j < x.m_dim_y) {
        // //// fp16 version
        // half var = 0;

        // for (int k = 0; k < x.m_dim_z; k++) {  // hidden states
        //     // var = __hadd(var, __hmul(x(i, j, k), x(i, j, k)));
        //     var = __hfma(x(i, j, k), x(i, j, k), var);
        // }

        // var = __hdiv(var, __int2half_rn(x.m_dim_z));
        // half variance = hrcp(hsqrt(__hadd(var, eps)));
        // // half variance = __float2half(1.0f / sqrtf(__half2float(var) + eps));

        // for (int k = 0; k < x.m_dim_z; k++) {
        //     half value = x(i, j, k);
        //     // half half_out = __hmul(__hmul(value, variance), __float2half(weight(0, 0, k)));  // TODO: convert weight to half
        //     half half_out = __float2half(__half2float(__hmul(value, variance)) * weight(0, 0, k));  // TODO: convert weight to half
            
        //     output(i, j, k) = half_out;
        // }


        //// fp32 version
        float var = 0.0f;

        for (int k = 0; k < x.m_dim_z; k++) {  // hidden states
            float value = __half2float(x(i, j, k));
            var += value * value;
        }

        var /= static_cast<float>(x.m_dim_z);
        float variance = rsqrtf(var + eps);

        for (int k = 0; k < x.m_dim_z; k++) {
            float value = __half2float(x(i, j, k));
            float fp_out = (value * variance) * weight(0, 0, k);
            
            // output(i, j, k) = fp_out;
            output(i, j, k) = __float2half(fp_out);
        }


        // //// fp16 version
        // half var = 0;

        // for (int k = 0; k < x.m_dim_z; k++) {  // hidden states
        //     half value = x(i, j, k);
        //     var = __hfma(value, value, var);
        // }

        // var = __hdiv(var, __int2half_rn(x.m_dim_z));
        // half variance = hrcp(hsqrt(__hadd(var, epf_half)));

        // for (int k = 0; k < x.m_dim_z; k++) {
        //     half value = x(i, j, k);
        //     half half_out = __hmul(value, variance) * __float2half(weight(0, 0, k));
            
        //     // output(i, j, k) = fp_out;
        //     output(i, j, k) = half_out;
        // }
    }
}

// __global__ void LlamaRMSNorm_cuda_kernel(const Matrix3D<half> x, const Matrix3D<float> weight, Matrix3D<half> output, float eps) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     int k = blockIdx.z * blockDim.z + threadIdx.z;

//     // index into shared memory (assuming it's size is at least blockDim.z)
//     int shared_index = threadIdx.z;
//     float shared_memory[11008];

//     if (i < x.m_dim_x && j < x.m_dim_y && k < x.m_dim_z) {
//         float value = __half2float(x(i, j, k));
//         shared_memory[shared_index] = value * value;  // square of the value
//     }

//     __syncthreads();  // make sure all threads have written to shared memory

//     // now perform reduction in shared memory
//     for (unsigned int s = blockDim.z / 2; s > 0; s >>= 1) {
//         if (threadIdx.z < s) {
//             shared_memory[shared_index] += shared_memory[shared_index + s];
//         }
//         __syncthreads();  // make sure all additions are done before next step
//     }

//     if (threadIdx.z == 0) {  // only thread 0 writes the result
//         float var = shared_memory[0] / static_cast<float>(x.m_dim_z);
//         float variance = 1.0 / sqrtf(var + eps);

//         // now go through z dimension again to do normalization
//         for (int kk = 0; kk < x.m_dim_z; kk++) {
//             float value = __half2float(x(i, j, kk));
//             float fp_out = (value * variance) * weight(0, 0, kk);
//             output(i, j, kk) = __float2half(fp_out);
//         }
//     }
// }

void LlamaRMSNorm_cuda::forward(const Matrix3D<half> &x, Matrix3D<half> &output) {
    dim3 block(32, 32);
    dim3 grid((x.m_dim_x + block.x - 1) / block.x, (x.m_dim_y + block.y - 1) / block.y);

    LlamaRMSNorm_cuda_kernel<<<grid, block>>>(x, weight, output, eps);
}
