#include "common.h"

class BMM_F16T{
public:
    BMM_F16T(half _alpha);
    BMM_F16T(){};
    void forward(const Matrix3D<half> &x, const Matrix3D<half> &weight, Matrix3D<half> &output);  // TODO: convert weight to half
    void forward_weight_untransposed(const Matrix3D<half> &a, const Matrix3D<half> &weight, Matrix3D<half> &c);
    half alpha;
private:
    std::string profile_name = "BMM_F16T";
};

void load_BMM_F16T(BMM_F16T &op, std::string prefix);
