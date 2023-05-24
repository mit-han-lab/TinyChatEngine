#include "common.h"

struct W8A8BFP32OFP32Linear_params {
    Matrix3D<int8_t> weight;
    Matrix3D<float> bias;
    float alpha;
};


class W8A8BFP32OFP32Linear{
public:
    W8A8BFP32OFP32Linear(W8A8BFP32OFP32Linear_params &params_);
    W8A8BFP32OFP32Linear(){};
    void forward(const Matrix3D<int8_t> &x, Matrix3D<float> &output);
    struct matmul_params params;
    float alpha;
private:
    std::string profile_name = "W8A8BFP32OFP32Linear";
};

void load_W8A8BFP32OFP32Linear_params(W8A8BFP32OFP32Linear &op, std::string prefix);