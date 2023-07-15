#include "common.cuh"
#include <cstdlib>
#include "utils.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

class RotaryPosEmb_half
{
public:
    RotaryPosEmb_half(Matrix3D_cuda<float> _cos, Matrix3D_cuda<float> _sin, std::string path)
    {
        sin = _sin;
        cos = _cos;
        read_to_array((path + "/cos_cached.bin").c_str(), cos.m_data, cos.length());
        read_to_array((path + "/sin_cached.bin").c_str(), sin.m_data, sin.length());
    };
    RotaryPosEmb_half(){};
    Matrix3D_cuda<float> cos, sin;

private:
    std::string profile_name = "RotaryPosEmb_half";
};

void load_RotaryPosEmb_half(RotaryPosEmb_half &op, std::string prefix);

__global__ void RotaryPosEmb_half_forward(Matrix3D_cuda<float> query, Matrix3D_cuda<float> key, Matrix3D_cuda<float> cos, Matrix3D_cuda<float> sin, int start_idx, int len);
