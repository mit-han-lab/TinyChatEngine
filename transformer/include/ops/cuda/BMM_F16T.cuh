#include "common.h"
// #include "common.cuh"

class BMM_F16T{
public:
    BMM_F16T(float _alpha);
    BMM_F16T(){};
    void forward(const Matrix3D<float> &x, const Matrix3D<float> &weight, Matrix3D<float> &output);
    float alpha;
private:
    std::string profile_name = "BMM_F16T";
};

void load_BMM_F16T(BMM_F16T &op, std::string prefix);

// __global__ void BMM_F16T_forward(const Matrix3D<float> a, const Matrix3D<float> weight, Matrix3D<float> c);
