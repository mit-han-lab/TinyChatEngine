#include <cstdlib>

#include "utils.h"
#include "common.h"

class RotaryPosEmb_metal
{
public:
    RotaryPosEmb_metal(Matrix3D<half> _cos, Matrix3D<half> _sin, std::string path)
    {
        sin = _sin;
        cos = _cos;
        read_to_array_half((path + "/cos_cached_half.bin").c_str(), cos.m_data, cos.length());
        read_to_array_half((path + "/sin_cached_half.bin").c_str(), sin.m_data, sin.length());
    };
    RotaryPosEmb_metal(){};
    void forward(Matrix3D<half> &key, Matrix3D<half> &value, int start_idx, int len);
    Matrix3D<half> cos, sin;

private:
    std::string profile_name = "RotaryPosEmb_metal";
};

void load_RotaryPosEmb_metal(RotaryPosEmb_metal &op, std::string prefix);

void RotaryPosEmb_metal_forward(Matrix3D<half> query, Matrix3D<half> key, Matrix3D<half> cos, Matrix3D<half> sin, int start_idx, int len);
