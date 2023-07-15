#include "common.cuh"
#include "common.h"

class BMM_F16T{
public:
    BMM_F16T(float _alpha);
    BMM_F16T(){};
    void forward(const Matrix3D_cuda<float> &x, const Matrix3D_cuda<float> &weight, Matrix3D<float> &output);
    float alpha;
private:
    std::string profile_name = "BMM_F16T";
};

void load_BMM_F16T(BMM_F16T &op, std::string prefix);