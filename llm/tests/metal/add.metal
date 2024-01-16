/*
A kernel that adds two arrays of floats.
*/

#include <metal_stdlib>
#include "common.h"
using namespace metal;
// Assume that inputA, inputB, and output have the same dimension
kernel void elementwise_add(
    device const Matrix3D<float> &inputA,
    device const Matrix3D<float> &inputB,
    device const Matrix3D<float> &output,
    uint3 gid [[ thread_position_in_grid ]]
)
{
    // // Calculate the global linear index for Matrix3D
    // uint index = gid.x + gid.y * inputA.m_dim_y + gid.z * inputA.m_dim_y * inputA.m_dim_z;

    // Perform element-wise addition
    output(gid.x, gid.y, gid.z) = inputA(gid.x, gid.y, gid.z) + inputB(gid.x, gid.y, gid.z);
}
