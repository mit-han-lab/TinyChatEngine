#include <cstdlib>
#include "utils.h"
#include "common.h"

class RotaryPosEmb_half
{
public:
    RotaryPosEmb_half(Matrix3D<float> _cos, Matrix3D<float> _sin, std::string path)
    {
        sin = _sin;
        cos = _cos;
        read_to_array((path + "/cos_cached.bin").c_str(), cos.m_data, cos.length());
        read_to_array((path + "/sin_cached.bin").c_str(), sin.m_data, sin.length());
    };
    RotaryPosEmb_half(){};
    void forward(Matrix3D<half> &key, Matrix3D<half> &value, int start_idx, int len);
    Matrix3D<float> cos, sin;

private:
    std::string profile_name = "RotaryPosEmb_half";
};

void load_RotaryPosEmb_half(RotaryPosEmb_half &op, std::string prefix);

__global__ void RotaryPosEmb_float_forward(Matrix3D<float> query, Matrix3D<float> key, Matrix3D<float> cos, Matrix3D<float> sin, int start_idx, int len);
__global__ void RotaryPosEmb_half_forward(Matrix3D<half> query, Matrix3D<half> key, Matrix3D<float> cos, Matrix3D<float> sin, int start_idx, int len);
