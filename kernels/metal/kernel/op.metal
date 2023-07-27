#include <metal_stdlib>
using namespace metal;

/*
Performance comparision with the test case:
CPU: 4000ms, ~60GOP/s
matmulInt4: 8000ms, ~30GOP/s
matmulInt4_SIMD_Q4Interleave(GPU): 3000 ms, ~83 GOP/s
matmulInt4_SIMD_Q4Interleave_unroll16(GPU): 1800 ms, 133 GOP/s
matmulInt4_SIMD_Q4Interleave_unroll32(GPU): 1500 ms, 160 GOP/s
*/

#include "opParams.h"
kernel void matmul(device const float* inA,
                    device const float* inB, // column major
                    device float* result,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    float sum = 0;
    for (uint i = 0; i < k; i++){
        float vA = inA[idy * k + i];
        float vB = inB[idx * k + i];

        sum += vA * vB;
    }
    result[idy * n + idx] = sum;
}

kernel void matmulInt4(device const float* inA,
                    device const uint8_t* inB, // column major
                    device float* result,
                    device const float* scales,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;
    const uint group_size = params.group_size;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    float sum = 0;
    for (uint i = 0; i < k; i += group_size){
        float scale = scales[(idx * k + i) / group_size];
        for (uint j = 0; j < group_size; j+=2){
            size_t weight_idx = (idx * k + i + j) / 2;
            uint8_t weight_packed = inB[weight_idx];
            int8_t vl = (weight_packed & 0x0F) - 8;
            int8_t vh = (weight_packed >> 4) - 8;

            sum += (inA[idy * k + i + j] * vl) * scale;
            sum += (inA[idy * k + i + j + 1] * vh) * scale;
        }
    }
    result[idy * n + idx] = sum;
}


kernel void matmulInt4_SIMD_Q4Interleave(
                    device const packed_float4* inA,
                    device const packed_char4* inB, // column major
                    device float* result,
                    device const float* scales,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;
    const uint group_size = params.group_size;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    packed_char4 lowMask = {0x0F, 0x0F, 0x0F, 0x0F};
    packed_float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = 0; i < k; i += group_size){
        float scale = scales[(idx * k + i) / group_size];
        packed_float4 scale4 = {scale, scale, scale, scale};
        for (uint j = 0; j < group_size; j+= 8){
            // sequential: (a, b), (c, d), (e, f), (g, h): 32 bit = 4xuint8
            // expected layout of inB: (a, e), (b, f), (c, g), (d, h)
            // low; (a, 0), (b, 0), (c, 0), (d, 0)
            // high: (e, 0), (f, 0), (g, 0), (h, 0)
            size_t weight_idx = (idx * k + i + j) / 8;
            size_t activation_idx = (idy * k + i + j) / 4;
            packed_char4 packed_8 = inB[weight_idx];
            packed_char4 packed_low = packed_8 & lowMask;
            packed_char4 packed_high = (packed_8 >> 4) & lowMask;

            packed_float4 inAlow = inA[activation_idx];
            packed_float4 inAhigh = inA[activation_idx+1];
            packed_float4 inBlow = packed_float4(packed_low) * scale4;
            packed_float4 inBhigh = packed_float4(packed_high) * scale4;

            sum4 += inAlow * inBlow;
            sum4 += inAhigh * inBhigh;
        }
    }
    float sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];
    result[idy * n + idx] = sum;
}

kernel void matmulUInt4_SIMD_Q4Interleave_unroll16(
                    device const packed_float4* inA,
                    device const packed_char4* inB, // column major
                    device float* result,
                    device const float* scales,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;
    const uint group_size = params.group_size;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    packed_char4 lowMask = {0x0F, 0x0F, 0x0F, 0x0F};
    packed_float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};
    packed_char4 offsets = {8, 8, 8, 8};

    for (uint i = 0; i < k; i += group_size){
        float scale = scales[(idx * k + i) / group_size];
        packed_float4 scale4 = {scale, scale, scale, scale};
        for (uint j = 0; j < group_size; j+= 16){
            // sequential: (a, b), (c, d), (e, f), (g, h): 32 bit = 4xuint8
            // expected layout of inB: (a, e), (b, f), (c, g), (d, h)
            // low; (a, 0), (b, 0), (c, 0), (d, 0)
            // high: (e, 0), (f, 0), (g, 0), (h, 0)
            size_t weight_idx = (idx * k + i + j) / 8;
            size_t activation_idx = (idy * k + i + j) / 4;
            packed_char4 packed_8_0 = inB[weight_idx];
            packed_char4 packed_8_1 = inB[weight_idx + 1];
            packed_char4 packed_low_0 = (packed_8_0 & lowMask) - offsets;;
            packed_char4 packed_low_1 = (packed_8_1 & lowMask) - offsets;;
            packed_char4 packed_high_0 = ((packed_8_0 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_1 = ((packed_8_1 >> 4) & lowMask) - offsets;

            packed_float4 inAlow_0 = inA[activation_idx];
            packed_float4 inAlow_1 = inA[activation_idx+2];
            packed_float4 inAhigh_0 = inA[activation_idx+1];
            packed_float4 inAhigh_1 = inA[activation_idx+3];
            packed_float4 inBlow_0 = packed_float4(packed_low_0) * scale4;
            packed_float4 inBlow_1 = packed_float4(packed_low_1) * scale4;
            packed_float4 inBhigh_0 = packed_float4(packed_high_0) * scale4;
            packed_float4 inBhigh_1 = packed_float4(packed_high_1) * scale4;

            sum4 += inAlow_0 * inBlow_0;
            sum4 += inAlow_1 * inBlow_1;
            sum4 += inAhigh_0 * inBhigh_0;
            sum4 += inAhigh_1 * inBhigh_1;
        }
    }
    float sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];
    result[idy * n + idx] = sum;
}


kernel void matmulUInt4_SIMD_Q4Interleave_unroll32(
                    device const packed_float4* inA,
                    device const packed_char4* inB, // column major
                    device float* result,
                    device const float* scales,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;
    const uint group_size = params.group_size;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    packed_char4 lowMask = {0x0F, 0x0F, 0x0F, 0x0F};
    packed_char4 offsets = {8, 8, 8, 8};
    packed_float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = 0; i < k; i += group_size){
        float scale = scales[(idx * k + i) / group_size];
        packed_float4 scale4 = {scale, scale, scale, scale};
        for (uint j = 0; j < group_size; j+= 32){
            // sequential: (a, b), (c, d), (e, f), (g, h): 32 bit = 4xuint8
            // expected layout of inB: (a, e), (b, f), (c, g), (d, h)
            // low; (a, 0), (b, 0), (c, 0), (d, 0)
            // high: (e, 0), (f, 0), (g, 0), (h, 0)
            size_t weight_idx = (idx * k + i + j) / 8;
            size_t activation_idx = (idy * k + i + j) / 4;
            packed_char4 packed_8_0 = inB[weight_idx];
            packed_char4 packed_8_1 = inB[weight_idx + 1];
            packed_char4 packed_8_2 = inB[weight_idx + 2];
            packed_char4 packed_8_3 = inB[weight_idx + 3];

            packed_char4 packed_low_0 = (packed_8_0 & lowMask) - offsets;;
            packed_char4 packed_low_1 = (packed_8_1 & lowMask) - offsets;;
            packed_char4 packed_low_2 = (packed_8_2 & lowMask) - offsets;;
            packed_char4 packed_low_3 = (packed_8_3 & lowMask) - offsets;;

            packed_char4 packed_high_0 = ((packed_8_0 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_1 = ((packed_8_1 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_2 = ((packed_8_2 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_3 = ((packed_8_3 >> 4) & lowMask) - offsets;

            packed_float4 inAlow_0 = inA[activation_idx];
            packed_float4 inAhigh_0 = inA[activation_idx+1];
            packed_float4 inAlow_1 = inA[activation_idx+2];
            packed_float4 inAhigh_1 = inA[activation_idx+3];
            packed_float4 inAlow_2 = inA[activation_idx+4];
            packed_float4 inAhigh_2 = inA[activation_idx+5];
            packed_float4 inAlow_3 = inA[activation_idx+6];
            packed_float4 inAhigh_3 = inA[activation_idx+7];

            packed_float4 inBlow_0 = packed_float4(packed_low_0) * scale4;
            packed_float4 inBlow_1 = packed_float4(packed_low_1) * scale4;
            packed_float4 inBlow_2 = packed_float4(packed_low_2) * scale4;
            packed_float4 inBlow_3 = packed_float4(packed_low_3) * scale4;

            packed_float4 inBhigh_0 = packed_float4(packed_high_0) * scale4;
            packed_float4 inBhigh_1 = packed_float4(packed_high_1) * scale4;
            packed_float4 inBhigh_2 = packed_float4(packed_high_2) * scale4;
            packed_float4 inBhigh_3 = packed_float4(packed_high_3) * scale4;

            sum4 += inAlow_0 * inBlow_0;
            sum4 += inAlow_1 * inBlow_1;
            sum4 += inAlow_2 * inBlow_2;
            sum4 += inAlow_3 * inBlow_3;
            sum4 += inAhigh_0 * inBhigh_0;
            sum4 += inAhigh_1 * inBhigh_1;
            sum4 += inAhigh_2 * inBhigh_2;
            sum4 += inAhigh_3 * inBhigh_3;
        }
    }
    float sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];
    result[idy * n + idx] = sum;
}

kernel void matmulUInt4_SIMD_Q4Interleave_unroll2x32(
                    device const packed_float4* inA,
                    device const packed_char4* inB, // column major
                    device float* result,
                    device const float* scales,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;
    const uint group_size = params.group_size;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    packed_char4 lowMask = {0x0F, 0x0F, 0x0F, 0x0F};
    packed_char4 offsets = {8, 8, 8, 8};
    packed_float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};
    packed_float4 sum4_col2 = {0.0f, 0.0f, 0.0f, 0.0f};

    packed_float4 a;

    for (uint i = 0; i < k; i += group_size){
        float scale = scales[(idx * k + i) / group_size];
        float scale_col2 = scales[((idx+1) * k + i) / group_size];
        packed_float4 scale4 = {scale, scale, scale, scale};
        packed_float4 scale4_col2 = {scale_col2, scale_col2, scale_col2, scale_col2};
        for (uint j = 0; j < group_size; j+= 32){
            // sequential: (a, b), (c, d), (e, f), (g, h): 32 bit = 4xuint8
            // expected layout of inB: (a, e), (b, f), (c, g), (d, h)
            // low; (a, 0), (b, 0), (c, 0), (d, 0)
            // high: (e, 0), (f, 0), (g, 0), (h, 0)
            size_t weight_idx = (idx * k + i + j) / 8;
            size_t weight_col2_idx = ((idx+1) * k + i + j) / 8;
            size_t activation_idx = (idy * k + i + j) / 4;
            packed_char4 packed_8_0 = inB[weight_idx];
            packed_char4 packed_8_1 = inB[weight_idx + 1];
            packed_char4 packed_8_2 = inB[weight_idx + 2];
            packed_char4 packed_8_3 = inB[weight_idx + 3];

            packed_char4 packed_low_0 = (packed_8_0 & lowMask) - offsets;
            packed_char4 packed_low_1 = (packed_8_1 & lowMask) - offsets;
            packed_char4 packed_low_2 = (packed_8_2 & lowMask) - offsets;
            packed_char4 packed_low_3 = (packed_8_3 & lowMask) - offsets;

            packed_char4 packed_high_0 = ((packed_8_0 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_1 = ((packed_8_1 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_2 = ((packed_8_2 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_3 = ((packed_8_3 >> 4) & lowMask) - offsets;

            packed_float4 inAlow_0 = inA[activation_idx];
            packed_float4 inAhigh_0 = inA[activation_idx+1];
            packed_float4 inAlow_1 = inA[activation_idx+2];
            packed_float4 inAhigh_1 = inA[activation_idx+3];
            packed_float4 inAlow_2 = inA[activation_idx+4];
            packed_float4 inAhigh_2 = inA[activation_idx+5];
            packed_float4 inAlow_3 = inA[activation_idx+6];
            packed_float4 inAhigh_3 = inA[activation_idx+7];

            packed_float4 inBlow_0 = packed_float4(packed_low_0) * scale4;
            packed_float4 inBlow_1 = packed_float4(packed_low_1) * scale4;
            packed_float4 inBlow_2 = packed_float4(packed_low_2) * scale4;
            packed_float4 inBlow_3 = packed_float4(packed_low_3) * scale4;

            packed_float4 inBhigh_0 = packed_float4(packed_high_0) * scale4;
            packed_float4 inBhigh_1 = packed_float4(packed_high_1) * scale4;
            packed_float4 inBhigh_2 = packed_float4(packed_high_2) * scale4;
            packed_float4 inBhigh_3 = packed_float4(packed_high_3) * scale4;

            sum4 += inAlow_0 * inBlow_0;
            sum4 += inAlow_1 * inBlow_1;
            sum4 += inAlow_2 * inBlow_2;
            sum4 += inAlow_3 * inBlow_3;
            sum4 += inAhigh_0 * inBhigh_0;
            sum4 += inAhigh_1 * inBhigh_1;
            sum4 += inAhigh_2 * inBhigh_2;
            sum4 += inAhigh_3 * inBhigh_3;

        }
    }
    float sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];
    result[idy * n + idx] = sum;
}

kernel void matmulUInt4_SIMD_Q4Interleave_half_unroll32(
                    device const packed_half4* inA,
                    device const packed_char4* inB, // column major
                    device float* result,
                    device const float* scales,
                    constant MetalMatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint n = params.n;
    const uint k = params.k;
    const uint group_size = params.group_size;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    packed_char4 lowMask = {0x0F, 0x0F, 0x0F, 0x0F};
    packed_char4 offsets = {8, 8, 8, 8};
    packed_half4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint i = 0; i < k; i += group_size){
        half scale = half(scales[(idx * k + i) / group_size]);
        packed_half4 scale4 = {scale, scale, scale, scale};
        for (uint j = 0; j < group_size; j+= 32){
            // sequential: (a, b), (c, d), (e, f), (g, h): 32 bit = 4xuint8
            // expected layout of inB: (a, e), (b, f), (c, g), (d, h)
            // low; (a, 0), (b, 0), (c, 0), (d, 0)
            // high: (e, 0), (f, 0), (g, 0), (h, 0)
            size_t weight_idx = (idx * k + i + j) / 8;
            size_t activation_idx = (idy * k + i + j) / 4;
            packed_char4 packed_8_0 = inB[weight_idx];
            packed_char4 packed_8_1 = inB[weight_idx + 1];
            packed_char4 packed_8_2 = inB[weight_idx + 2];
            packed_char4 packed_8_3 = inB[weight_idx + 3];

            packed_char4 packed_low_0 = (packed_8_0 & lowMask) - offsets;;
            packed_char4 packed_low_1 = (packed_8_1 & lowMask) - offsets;;
            packed_char4 packed_low_2 = (packed_8_2 & lowMask) - offsets;;
            packed_char4 packed_low_3 = (packed_8_3 & lowMask) - offsets;;

            packed_char4 packed_high_0 = ((packed_8_0 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_1 = ((packed_8_1 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_2 = ((packed_8_2 >> 4) & lowMask) - offsets;
            packed_char4 packed_high_3 = ((packed_8_3 >> 4) & lowMask) - offsets;

            packed_half4 inAlow_0 = inA[activation_idx];
            packed_half4 inAhigh_0 = inA[activation_idx+1];
            packed_half4 inAlow_1 = inA[activation_idx+2];
            packed_half4 inAhigh_1 = inA[activation_idx+3];
            packed_half4 inAlow_2 = inA[activation_idx+4];
            packed_half4 inAhigh_2 = inA[activation_idx+5];
            packed_half4 inAlow_3 = inA[activation_idx+6];
            packed_half4 inAhigh_3 = inA[activation_idx+7];

            packed_half4 inBlow_0 = packed_half4(packed_low_0) * scale4;
            packed_half4 inBlow_1 = packed_half4(packed_low_1) * scale4;
            packed_half4 inBlow_2 = packed_half4(packed_low_2) * scale4;
            packed_half4 inBlow_3 = packed_half4(packed_low_3) * scale4;

            packed_half4 inBhigh_0 = packed_half4(packed_high_0) * scale4;
            packed_half4 inBhigh_1 = packed_half4(packed_high_1) * scale4;
            packed_half4 inBhigh_2 = packed_half4(packed_high_2) * scale4;
            packed_half4 inBhigh_3 = packed_half4(packed_high_3) * scale4;

            sum4 += inAlow_0 * inBlow_0;
            sum4 += inAlow_1 * inBlow_1;
            sum4 += inAlow_2 * inBlow_2;
            sum4 += inAlow_3 * inBlow_3;
            sum4 += inAhigh_0 * inBhigh_0;
            sum4 += inAhigh_1 * inBhigh_1;
            sum4 += inAhigh_2 * inBhigh_2;
            sum4 += inAhigh_3 * inBhigh_3;
        }
    }
    half sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];
    result[idy * n + idx] = float(sum);
}
