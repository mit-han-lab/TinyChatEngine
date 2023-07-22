#include "common.h"
// #include "common.cuh"

class BMM_F16T{
public:
    BMM_F16T(float _alpha);  // TODO: convert alpha to half
    BMM_F16T(){};
    void forward(const Matrix3D<half> &x, const Matrix3D<half> &weight, Matrix3D<half> &output);  // TODO: convert weight to half
    float alpha;
private:
    std::string profile_name = "BMM_F16T";
};

void load_BMM_F16T(BMM_F16T &op, std::string prefix);
