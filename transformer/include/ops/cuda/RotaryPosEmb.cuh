#include <cstdlib>

#include "utils.h"
#include "common.h"

class RotaryPosEmb_cuda
{
public:
    RotaryPosEmb_cuda(Matrix3D<half> _cos, Matrix3D<half> _sin, std::string path)
    {
        sin = _sin;
        cos = _cos;
        read_to_array_half((path + "/cos_cached_half.bin").c_str(), cos.m_data, cos.length());
        read_to_array_half((path + "/sin_cached_half.bin").c_str(), sin.m_data, sin.length());
    };
    RotaryPosEmb_cuda(){};
    void forward(Matrix3D<half> &key, Matrix3D<half> &value, int start_idx, int len);
    Matrix3D<half> cos, sin;

private:
    std::string profile_name = "RotaryPosEmb_cuda";
};

void load_RotaryPosEmb_cuda(RotaryPosEmb_cuda &op, std::string prefix);

__global__ void RotaryPosEmb_float_forward(Matrix3D<float> query, Matrix3D<float> key, Matrix3D<float> cos, Matrix3D<float> sin, int start_idx, int len);
__global__ void RotaryPosEmb_cuda_forward(Matrix3D<half> query, Matrix3D<half> key, Matrix3D<half> cos, Matrix3D<half> sin, int start_idx, int len);
