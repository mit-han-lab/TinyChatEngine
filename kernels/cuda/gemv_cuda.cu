
// Adapted from https://github.com/mit-han-lab/llm-awq
/*

@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}

*/

#include <assert.h>
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <stdio.h>

#include "../matmul.h"
#include "ops/linear.h"

// #include <cuda_runtime.h>
#include <cuda_fp16.h>

#define VECTORIZE_FACTOR 8
#define Q_VECTORIZE_FACTOR 8
#define PACK_FACTOR 8
#define WARP_SIZE 32


// Reduce sum within the warp using the tree reduction algorithm.
__device__ __forceinline__ float warp_reduce_sum(float sum) {
  #pragma unroll
  for(int i = 4; i >= 0; i--){
    sum += __shfl_down_sync(0xffffffff, sum, 1<<i);
  }
  /*
  // Equivalent to the following tree reduction implementation:
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);
  */
  return sum;
}

__device__ __forceinline__ int make_divisible(int c, int divisor){
  return (c + divisor - 1) / divisor;
}


/*
Computes GEMV (group_size = 64).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC / 8];
  output: vector of shape [OC];
  zeros: matrix of shape [OC, IC / group_size / 8];
  scaling_factors: matrix of shape [OC, IC / group_size];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void gemv_kernel_g64(
  const float4* _inputs, const uint32_t* weight, const uint32_t* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 64;
    float psum = 0;
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
    half* outputs = _outputs + batch_idx * OC;
    // This is essentially zeros_w.
    const int num_groups_packed = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2;
    const int weight_w = IC / PACK_FACTOR;
    // TODO (Haotian): zeros_w is incorrect, after fixing we got misaligned address
    const int zeros_w = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2;
    // consistent with input shape
    const int sf_w = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2 * PACK_FACTOR;
    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d %d %d %d\n", IC, group_size, PACK_FACTOR, zeros_w, sf_w);
    // tile size: 4 OC x 1024 IC per iter
    for(int packed_group_idx = 0; packed_group_idx < num_groups_packed / 2; packed_group_idx++){
      // 1024 numbers in one iteration across warp. Need 1024 / group_size zeros.
      uint64_t packed_zeros = *reinterpret_cast<const uint64_t*>(zeros + oc_idx * zeros_w + packed_group_idx * 2);
      uint32_t packed_weights[4];
      // use float4 to load weights, each thread load 32 int4 numbers (1 x float4)
      *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w + packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));
      // load scaling factors
      // g64: two threads -> 64 numbers -> 1 group; 1 warp = 16 groups.
      float scaling_factor = __half2float(scaling_factors[oc_idx * sf_w + packed_group_idx * 16 + (threadIdx.x / 2)]);
      float current_zeros = (float)((packed_zeros >> (threadIdx.x / 2 * 4)) & 0xF);
      int inputs_ptr_delta = packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4; 
      const float4* inputs_ptr = inputs + inputs_ptr_delta;
      // multiply 32 weights with 32 inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        // iterate over different uint32_t packed_weights in this loop
        uint32_t current_packed_weight = packed_weights[ic_0];
        half packed_inputs[PACK_FACTOR];
        // each thread load 8 inputs, starting index is packed_group_idx * 128 * 8 (because each iter loads 128*8)
        if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
          *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
          #pragma unroll
          for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
            // iterate over 8 numbers packed within each uint32_t number
            float current_single_weight_fp = (float)(current_packed_weight & 0xF);
            float dequantized_weight = scaling_factor * (current_single_weight_fp - current_zeros);
            //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0) printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
            psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
            current_packed_weight = current_packed_weight >> 4;
          }
        }
      }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
     outputs[oc_idx] = __float2half(psum); 
    }
}


/*
Computes GEMV (group_size = 128).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC / 8];
  output: vector of shape [OC];
  zeros: matrix of shape [OC, IC / group_size / 8];
  scaling_factors: matrix of shape [OC, IC / group_size];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void gemv_kernel_g128(
  const float4* _inputs, const uint32_t* weight, const uint32_t* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 128;
    float psum = 0;
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
    half* outputs = _outputs + batch_idx * OC;
    const int num_groups_packed = make_divisible(IC / group_size, PACK_FACTOR);
    const int weight_w = IC / PACK_FACTOR;
    // TODO (Haotian): zeros_w is incorrect, after fixing we got misaligned address
    const int zeros_w = make_divisible(IC / group_size, PACK_FACTOR);
    // consistent with input shape
    const int sf_w = make_divisible(IC / group_size, PACK_FACTOR) * PACK_FACTOR;
    //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d %d %d\n", IC, group_size, PACK_FACTOR, zeros_w);
    // tile size: 4 OC x 1024 IC per iter
    for(int packed_group_idx = 0; packed_group_idx < num_groups_packed; packed_group_idx++){
      // 1024 numbers in one iteration across warp. Need 1024 / group_size zeros.
      uint32_t packed_zeros = *(zeros + oc_idx * zeros_w + packed_group_idx);
      uint32_t packed_weights[4];
      // use float4 to load weights, each thread load 32 int4 numbers (1 x float4)
      *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w + packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));
      // load scaling factors
      // g128: four threads -> 128 numbers -> 1 group; 1 warp = 8 groups.
      float scaling_factor = __half2float(scaling_factors[oc_idx * sf_w + packed_group_idx * 8 + (threadIdx.x / 4)]);
      float current_zeros = (float)((packed_zeros >> (threadIdx.x / 4 * 4)) & 0xF);
      int inputs_ptr_delta = packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4; 
      const float4* inputs_ptr = inputs + inputs_ptr_delta;
      // multiply 32 weights with 32 inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        // iterate over different uint32_t packed_weights in this loop
        uint32_t current_packed_weight = packed_weights[ic_0];
        half packed_inputs[PACK_FACTOR];
        // each thread load 8 inputs, starting index is packed_group_idx * 128 * 8 (because each iter loads 128*8)
        if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
          *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
          #pragma unroll
          for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
            // iterate over 8 numbers packed within each uint32_t number
            float current_single_weight_fp = (float)(current_packed_weight & 0xF);
            float dequantized_weight = scaling_factor * (current_single_weight_fp - current_zeros);
            //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0) printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
            psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
            current_packed_weight = current_packed_weight >> 4;
          }
        }
      }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
     outputs[oc_idx] = __float2half(psum); 
    }
}
/*
 * V1:
 *   1. Block 内 128 个线程协作、连续读取 activation
 *   2. activation 缓存到 Shared Memory
 *   3. 4 个 Warp 复用同一份 activation
 *   4. 使用 padding=1 降低 Shared Memory Bank Conflict
 */
__global__ void gemv_kernel_g128_v1(
    const float4* _inputs,
    const uint32_t* weight,
    const uint32_t* zeros,
    const half* scaling_factors,
    half* _outputs,
    const int IC,
    const int OC) {

    const int group_size = 128;

    float psum = 0.0f;

    const int batch_idx = blockIdx.z;
    const int oc_idx =
        blockIdx.y * blockDim.y + threadIdx.y;

    half* outputs =
        _outputs + batch_idx * OC;

    const int num_groups_packed =
        make_divisible(IC / group_size, PACK_FACTOR);

    const int weight_w =
        IC / PACK_FACTOR;

    const int zeros_w =
        make_divisible(IC / group_size, PACK_FACTOR);

    const int sf_w =
        make_divisible(IC / group_size, PACK_FACTOR)
        * PACK_FACTOR;


    // ============================================================
    // Shared Memory Layout
    //
    // 每个 lane 原本负责 32 个 FP16 activation
    //
    // 32 FP16 = 16 half2
    //
    // [32][17]：
    //
    // 第一维  = lane
    // 第二维  = 16 个 half2 + 1 个 padding
    //
    // padding 用于降低 bank conflict。
    //
    // 实际大小：
    // 32 * 17 * 4 B = 2176 B
    // ============================================================

    __shared__ half2 shared_inputs[WARP_SIZE][17];


    // ============================================================
    // 输入重新解释为 half2
    //
    // 一个 half2 = 2 个 FP16 = 4 Bytes
    // ============================================================

    const half2* input_half2 =
        reinterpret_cast<const half2*>(_inputs)
        + batch_idx * (IC / 2);


    // Block 共 128 threads
    const int linear_tid =
        threadIdx.y * blockDim.x + threadIdx.x;

    const int threads_per_block =
        blockDim.x * blockDim.y;


    // ============================================================
    // 每个 packed_group_idx 处理 1024 个 FP16 activation
    //
    // 1024 FP16
    // =
    // 512 half2
    // ============================================================

    constexpr int HALF2_PER_LANE = 16;
    constexpr int HALF2_PER_TILE =
        WARP_SIZE * HALF2_PER_LANE;   // 512


    for (int packed_group_idx = 0;
         packed_group_idx < num_groups_packed;
         packed_group_idx++) {


        // ========================================================
        // 1. 每个 Warp 对应 output channel 的量化权重
        // ========================================================

        uint32_t packed_zeros =
            *(zeros
              + oc_idx * zeros_w
              + packed_group_idx);


        uint32_t packed_weights[4];

        *((float4*)(packed_weights)) =
            *((float4*)(
                weight
                + oc_idx * weight_w
                + packed_group_idx * (WARP_SIZE * 4)
                + threadIdx.x * 4));


        // g128：
        // 4 threads 共用一组 scale / zero
        float scaling_factor =
            __half2float(
                scaling_factors[
                    oc_idx * sf_w
                    + packed_group_idx * 8
                    + threadIdx.x / 4
                ]
            );


        float current_zeros =
            static_cast<float>(
                (packed_zeros
                 >> ((threadIdx.x / 4) * 4))
                & 0xF
            );


        // ========================================================
        // 2. 128 threads 协作加载 activation
        //
        // 每个 Block：
        //
        // 128 threads
        // ×
        // 每线程循环 4 次
        //
        // = 512 half2
        // = 1024 FP16
        //
        // 对于每一次循环：
        //
        // warp lane0 → half2[p]
        // warp lane1 → half2[p+1]
        // ...
        //
        // 全局内存访问连续。
        // ========================================================

        const int tile_half2_base =
            packed_group_idx * HALF2_PER_TILE;

        #pragma unroll
        for (int load_iter = 0;
             load_iter < 4;
             load_iter++) {

            const int p =
                linear_tid
                + load_iter * threads_per_block;

            const int global_half2_idx =
                tile_half2_base + p;


            // p 对应原始布局中的：
            //
            // lane      = p / 16
            // pair_idx  = p % 16

            const int owner_lane =
                p / HALF2_PER_LANE;

            const int pair_idx =
                p % HALF2_PER_LANE;


            if (global_half2_idx < IC / 2) {

                shared_inputs[owner_lane][pair_idx] =
                    input_half2[global_half2_idx];

            }
            else {

                // 最后一个不完整 tile
                shared_inputs[owner_lane][pair_idx] =
                    __float2half2_rn(0.0f);
            }
        }


        // 所有 activation 已进入 Shared Memory
        __syncthreads();


        // ========================================================
        // 3. 每个 Warp 使用相同的 Shared Memory activation
        //
        // threadIdx.x 就是原始 lane
        //
        // lane 0:
        // input 0~31
        //
        // lane 1:
        // input 32~63
        //
        // ...
        // ========================================================

        const int inputs_ptr_delta =
            packed_group_idx * WARP_SIZE * 4
            + threadIdx.x * 4;


        #pragma unroll
        for (int ic_0 = 0;
             ic_0 < 4;
             ic_0++) {

            uint32_t current_packed_weight =
                packed_weights[ic_0];


            half packed_inputs[PACK_FACTOR];


            // 与原始 Kernel 保持一致的边界判断
            if (inputs_ptr_delta + ic_0
                < IC / PACK_FACTOR) {


                // 一个 uint32 weight 对应 8 个 input
                //
                // 8 FP16 = 4 half2
                //
                // Shared Memory 中：
                //
                // pair:
                // ic_0*4
                // ic_0*4+1
                // ic_0*4+2
                // ic_0*4+3

                half2* packed_inputs_half2 =
                    reinterpret_cast<half2*>(
                        packed_inputs
                    );


                #pragma unroll
                for (int pair = 0;
                     pair < 4;
                     pair++) {

                    packed_inputs_half2[pair] =
                        shared_inputs[
                            threadIdx.x
                        ][
                            ic_0 * 4 + pair
                        ];
                }


                // =================================================
                // 与原始 Kernel 完全相同的 INT4 反量化与计算
                // =================================================

                #pragma unroll
                for (int ic_1 = 0;
                     ic_1 < PACK_FACTOR;
                     ic_1++) {

                    float current_single_weight_fp =
                        static_cast<float>(
                            current_packed_weight
                            & 0xF
                        );


                    float dequantized_weight =
                        scaling_factor
                        * (
                            current_single_weight_fp
                            - current_zeros
                          );


                    psum +=
                        dequantized_weight
                        * __half2float(
                            packed_inputs[ic_1]
                          );


                    current_packed_weight >>= 4;
                }
            }
        }


        // ========================================================
        // 必须保证 4 个 Warp 都使用完当前 shared tile，
        // 才允许下一轮覆盖 shared_inputs。
        // ========================================================

        __syncthreads();
    }


    // ============================================================
    // Warp Reduction
    // ============================================================

    psum = warp_reduce_sum(psum);


    if (threadIdx.x == 0) {
        outputs[oc_idx] =
            __float2half(psum);
    }
}

__global__ void gemv_kernel_g128_v2(
    const float4* _inputs,
    const uint32_t* weight,
    const uint32_t* zeros,
    const half* scaling_factors,
    half* _outputs,
    const int IC,
    const int OC) {

    const int group_size = 128;

    float psum = 0.0f;

    const int batch_idx = blockIdx.z;
    const int oc_idx =
        blockIdx.y * blockDim.y + threadIdx.y;

    half* outputs =
        _outputs + batch_idx * OC;

    const int num_groups_packed =
        make_divisible(IC / group_size, PACK_FACTOR);

    const int weight_w =
        IC / PACK_FACTOR;

    const int zeros_w =
        make_divisible(IC / group_size, PACK_FACTOR);

    const int sf_w =
        make_divisible(IC / group_size, PACK_FACTOR)
        * PACK_FACTOR;


    // ============================================================
    // Shared Memory Layout
    //
    // 每个 lane 原本负责 32 个 FP16 activation
    //
    // 32 FP16 = 16 half2
    //
    // [32][17]：
    //
    // 第一维  = lane
    // 第二维  = 16 个 half2 + 1 个 padding
    //
    // padding 用于降低 bank conflict。
    //
    // 实际大小：
    // 32 * 17 * 4 B = 2176 B
    // ============================================================

    __shared__ half2 shared_inputs[WARP_SIZE][16];


    // ============================================================
    // 输入重新解释为 half2
    //
    // 一个 half2 = 2 个 FP16 = 4 Bytes
    // ============================================================

    const half2* input_half2 =
        reinterpret_cast<const half2*>(_inputs)
        + batch_idx * (IC / 2);


    // Block 共 128 threads
    const int linear_tid =
        threadIdx.y * blockDim.x + threadIdx.x;

    const int threads_per_block =
        blockDim.x * blockDim.y;


    // ============================================================
    // 每个 packed_group_idx 处理 1024 个 FP16 activation
    //
    // 1024 FP16
    // =
    // 512 half2
    // ============================================================

    constexpr int HALF2_PER_LANE = 16;
    constexpr int HALF2_PER_TILE =
        WARP_SIZE * HALF2_PER_LANE;   // 512


    for (int packed_group_idx = 0;
         packed_group_idx < num_groups_packed;
         packed_group_idx++) {


        // ========================================================
        // 1. 每个 Warp 对应 output channel 的量化权重
        // ========================================================

        uint32_t packed_zeros =
            *(zeros
              + oc_idx * zeros_w
              + packed_group_idx);


        uint32_t packed_weights[4];

        *((float4*)(packed_weights)) =
            *((float4*)(
                weight
                + oc_idx * weight_w
                + packed_group_idx * (WARP_SIZE * 4)
                + threadIdx.x * 4));


        // g128：
        // 4 threads 共用一组 scale / zero
        float scaling_factor =
            __half2float(
                scaling_factors[
                    oc_idx * sf_w
                    + packed_group_idx * 8
                    + threadIdx.x / 4
                ]
            );


        float current_zeros =
            static_cast<float>(
                (packed_zeros
                 >> ((threadIdx.x / 4) * 4))
                & 0xF
            );


        // ========================================================
        // 2. 128 threads 协作加载 activation
        //
        // 每个 Block：
        //
        // 128 threads
        // ×
        // 每线程循环 4 次
        //
        // = 512 half2
        // = 1024 FP16
        //
        // 对于每一次循环：
        //
        // warp lane0 → half2[p]
        // warp lane1 → half2[p+1]
        // ...
        //
        // 全局内存访问连续。
        // ========================================================

        const int tile_half2_base =
            packed_group_idx * HALF2_PER_TILE;

        #pragma unroll
        for (int load_iter = 0;
             load_iter < 4;
             load_iter++) {

            const int p =
                linear_tid
                + load_iter * threads_per_block;

            const int global_half2_idx =
                tile_half2_base + p;


            // p 对应原始布局中的：
            //
            // lane      = p / 16
            // pair_idx  = p % 16

            const int owner_lane =
                p / HALF2_PER_LANE;

            const int pair_idx =
                p % HALF2_PER_LANE;
            
            const int swizzled_pair_idx =
                pair_idx ^ (owner_lane >> 1);

            if (global_half2_idx < IC / 2) {

                shared_inputs[owner_lane][swizzled_pair_idx] =
                    input_half2[global_half2_idx];

            }
            else {

                // 最后一个不完整 tile
                shared_inputs[owner_lane][swizzled_pair_idx] =
                    __float2half2_rn(0.0f);
            }
        }


        // 所有 activation 已进入 Shared Memory
        __syncthreads();


        // ========================================================
        // 3. 每个 Warp 使用相同的 Shared Memory activation
        //
        // threadIdx.x 就是原始 lane
        //
        // lane 0:
        // input 0~31
        //
        // lane 1:
        // input 32~63
        //
        // ...
        // ========================================================

        const int inputs_ptr_delta =
            packed_group_idx * WARP_SIZE * 4
            + threadIdx.x * 4;


        #pragma unroll
        for (int ic_0 = 0;
             ic_0 < 4;
             ic_0++) {

            uint32_t current_packed_weight =
                packed_weights[ic_0];


            half packed_inputs[PACK_FACTOR];


            // 与原始 Kernel 保持一致的边界判断
            if (inputs_ptr_delta + ic_0
                < IC / PACK_FACTOR) {


                // 一个 uint32 weight 对应 8 个 input
                //
                // 8 FP16 = 4 half2
                //
                // Shared Memory 中：
                //
                // pair:
                // ic_0*4
                // ic_0*4+1
                // ic_0*4+2
                // ic_0*4+3

                half2* packed_inputs_half2 =
                    reinterpret_cast<half2*>(
                        packed_inputs
                    );


                #pragma unroll
                for (int pair = 0;
                     pair < 4;
                     pair++) {
                      const int logical_pair_idx =
                          ic_0 * 4 + pair;

                      const int swizzled_pair_idx =
                          logical_pair_idx ^ (threadIdx.x >> 1);

                      packed_inputs_half2[pair] =
                          shared_inputs[threadIdx.x][swizzled_pair_idx];
                }


                // =================================================
                // 与原始 Kernel 完全相同的 INT4 反量化与计算
                // =================================================

                #pragma unroll
                for (int ic_1 = 0;
                     ic_1 < PACK_FACTOR;
                     ic_1++) {

                    float current_single_weight_fp =
                        static_cast<float>(
                            current_packed_weight
                            & 0xF
                        );


                    float dequantized_weight =
                        scaling_factor
                        * (
                            current_single_weight_fp
                            - current_zeros
                          );


                    psum +=
                        dequantized_weight
                        * __half2float(
                            packed_inputs[ic_1]
                          );


                    current_packed_weight >>= 4;
                }
            }
        }


        // ========================================================
        // 必须保证 4 个 Warp 都使用完当前 shared tile，
        // 才允许下一轮覆盖 shared_inputs。
        // ========================================================

        __syncthreads();
    }


    // ============================================================
    // Warp Reduction
    // ============================================================

    psum = warp_reduce_sum(psum);


    if (threadIdx.x == 0) {
        outputs[oc_idx] =
            __float2half(psum);
    }
}


namespace matmul{
  
  /*
  Computes GEMV.

  Args:
    _in_feats: tensor of shape [B, IC];
    _kernel: int tensor of shape [OC, IC // 8];
    _zeros: int tensor of shape [OC, IC // G // 8];
    _scaling_factors: tensor of shape [OC, IC // G];
    blockDim_x: size of thread block, dimension x, where blockDim_x * workload_per_thread = IC;
    blockDim_y: size of thread block, dimension y, where blockDim_y * gridDim_y = OC;

  Returns:
    out_feats: tensor of shape [B, OC];
  */
  void MatmulOperator::gemv_forward_cuda(const struct matmul_params *params)
  {
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;

    int num_in_feats = A->row;
    int num_in_channels = A->column;
    int num_out_feats = C->row;
    int num_out_channels = C->column;
    int group_size = QK;

    auto in_feats = reinterpret_cast<float4*>(A->half_data_ptr);
    auto kernel = reinterpret_cast<uint32_t*>(B->int32_data_ptr);
    auto zeros = reinterpret_cast<uint32_t*>(params->int32_zero_point);
    auto scaling_factors = reinterpret_cast<half*>(params->half_scales);
    auto out_feats = reinterpret_cast<half*>(C->half_data_ptr);

    int blockDim_z = num_out_feats;
    dim3 num_blocks(1, num_out_channels / 4, num_out_feats);
    dim3 num_threads(32, 4);

    PROFILE_START("gemv_forward_cuda");

    if (group_size == 64)
    {
      gemv_kernel_g64<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, zeros, scaling_factors, out_feats,
        // constants
        num_in_channels, num_out_channels
      );
    }
    else if (group_size == 128)
    {
      gemv_kernel_g128<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, zeros, scaling_factors, out_feats,
        // constants
        num_in_channels, num_out_channels
      );
    }
    else
    {
      printf("Unsupported group size: %d\n", group_size);
      exit(1);
    }

    PROFILE_END("gemv_forward_cuda");
  }
  void MatmulOperator::gemv_forward_cuda_v1(
    const struct matmul_params* params) {

    const struct matrix* A = &params->A;
    const struct matrix* B = &params->B;
    const struct matrix* C = &params->C;

    int num_in_channels =
        A->column;

    int num_out_feats =
        C->row;

    int num_out_channels =
        C->column;

    int group_size =
        QK;


    auto in_feats =
        reinterpret_cast<float4*>(
            A->half_data_ptr
        );

    auto kernel =
        reinterpret_cast<uint32_t*>(
            B->int32_data_ptr
        );

    auto zeros =
        reinterpret_cast<uint32_t*>(
            params->int32_zero_point
        );

    auto scaling_factors =
        reinterpret_cast<half*>(
            params->half_scales
        );

    auto out_feats =
        reinterpret_cast<half*>(
            C->half_data_ptr
        );


    dim3 num_blocks(
        1,
        num_out_channels / 4,
        num_out_feats
    );

    dim3 num_threads(
        32,
        4
    );


    if (group_size == 128) {

        gemv_kernel_g128_v1<<<num_blocks, num_threads>>>(
                in_feats,
                kernel,
                zeros,
                scaling_factors,
                out_feats,
                num_in_channels,
                num_out_channels
            );

    }
    else {

        printf(
            "gemv_kernel_g128_v1 only supports "
            "group size 128\n"
        );

        exit(1);
    }
}
  void MatmulOperator::gemv_forward_cuda_v2(
    const struct matmul_params* params) {

    const struct matrix* A = &params->A;
    const struct matrix* B = &params->B;
    const struct matrix* C = &params->C;

    int num_in_channels =
        A->column;

    int num_out_feats =
        C->row;

    int num_out_channels =
        C->column;

    int group_size =
        QK;


    auto in_feats =
        reinterpret_cast<float4*>(
            A->half_data_ptr
        );

    auto kernel =
        reinterpret_cast<uint32_t*>(
            B->int32_data_ptr
        );

    auto zeros =
        reinterpret_cast<uint32_t*>(
            params->int32_zero_point
        );

    auto scaling_factors =
        reinterpret_cast<half*>(
            params->half_scales
        );

    auto out_feats =
        reinterpret_cast<half*>(
            C->half_data_ptr
        );


    dim3 num_blocks(
        1,
        num_out_channels / 4,
        num_out_feats
    );

    dim3 num_threads(
        32,
        4
    );


    if (group_size == 128) {

        gemv_kernel_g128_v2<<<num_blocks, num_threads>>>(
                in_feats,
                kernel,
                zeros,
                scaling_factors,
                out_feats,
                num_in_channels,
                num_out_channels
            );

    }
    else {

        printf(
            "gemv_kernel_g128_v2 only supports "
            "group size 128\n"
        );

        exit(1);
    }
}


  void MatmulOperator::mat_mul_accelerator_int4_fast(const struct matmul_params *params) {
    // TODO: remove this
  };

  void MatmulOperator::mat_mul_accelerator_int4_fast_no_offset(const struct matmul_params *params) {
    // TODO: remove this
  };

}  // namespace matmul

