#include <cmath>
#include <iomanip>

#include "operators.h"
#include "utils.h"
#include "reduction.cuh"

static inline __device__ float to_float(half src)
{
    return __half2float(src);
}

static inline __device__ half to_half(float src)
{
    return __float2half(src);
}


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
            float value = to_float(x(i, j, k));
            var += value * value;
        }

        var /= static_cast<float>(x.m_dim_z);
        float variance = rsqrtf(var + eps);

        for (int k = 0; k < x.m_dim_z; k++) {
            float value = to_float(x(i, j, k));
            float fp_out = (value * variance) * weight(0, 0, k);
            
            // output(i, j, k) = fp_out;
            output(i, j, k) = to_half(fp_out);
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


//// FasterTransformer version
__global__ void generalT5LayerNorm(
    const half* __restrict input, const float* __restrict gamma, half* output, const float layernorm_eps, int m, int n)
{
    // layernorm module in the T5 style No bias and no subtraction of mean.
    const int tid = threadIdx.x;

    __shared__ float s_variance;
    float            variance = 0.0f;

    float local_var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = to_float(__ldg(&input[blockIdx.x * n + i]));
        local_var_sum += diff * diff;
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)n + layernorm_eps);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        output[blockIdx.x * n + i] =
            clamp_inf_for_half<half>((to_float(input[blockIdx.x * n + i]) * s_variance) * __ldg(&gamma[i]));
    }
}


void LlamaRMSNorm_cuda::forward(const Matrix3D<half> &x, Matrix3D<half> &output) {
    //// Original version
    // dim3 block(32, 32);
    // dim3 grid((x.m_dim_x + block.x - 1) / block.x, (x.m_dim_y + block.y - 1) / block.y);
    // LlamaRMSNorm_cuda_kernel<<<grid, block>>>(x, weight, output, eps);
    

    //// FasterTransformer version
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

    /* should pay attention to the rsqrt precision*/
    half *input = x.m_data, *out = output.m_data;
    float *gamma = weight.m_data;
    generalT5LayerNorm<<<grid, block>>>(input, gamma, out, eps, m, n);  // For gpt-3
}
