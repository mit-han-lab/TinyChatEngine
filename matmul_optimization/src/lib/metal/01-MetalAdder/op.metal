/*
See LICENSE-original.txt for this sample’s licensing information.

Abstract:
A shader that adds two arrays of floats.
*/

#include <metal_stdlib>
using namespace metal;
/// This is a Metal Shading Language (MSL) function equivalent to the add_arrays()
/// C function, used to perform the calculation on a GPU.
kernel void add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] + inB[index];
}

#include "opParams.h"
kernel void matmul(device const float* inA,
                    device const float* inB, // column major
                    device float* result,
                    constant MatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint m = params.m;
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
                    constant MatMulParams& params,
                    uint2 id [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.

    const uint m = params.m;
    const uint n = params.n;
    const uint k = params.k;

    const uint idx = id.x; // column index of the output
    const uint idy = id.y; // row index of the output

    float sum = 0;
    for (uint i = 0; i < k; i+=2){
        float vA = inA[idy * k + i];
        float vB = inB[idx * k + i];

        size_t weight_idx = (idx * k + i) / 2;
        uint8_t weight_packed = inB[weight_idx];
        uint8_t vl = weight_packed & 0x0F;
        uint8_t vh = weight_packed >> 4;

        sum += inA[idy * k + i] * vl;
        sum += inA[idy * k + i + 1] * vh;
    }
    result[idy * n + idx] = sum;
}




