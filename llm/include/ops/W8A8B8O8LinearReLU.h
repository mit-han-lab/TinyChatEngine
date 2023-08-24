#include "common.h"

struct W8A8B8O8LinearReLU_params {
    Matrix3D<int8_t> weight;
    Matrix3D<int8_t> bias_int8;
    float alpha;
    float beta;
};

class W8A8B8O8LinearReLU {
   public:
    W8A8B8O8LinearReLU(W8A8B8O8LinearReLU_params &params_);
    W8A8B8O8LinearReLU(){};
    void forward(const Matrix3D<int8_t> &x, Matrix3D<int8_t> &output);
    struct matmul_params params;
    float alpha;
    float beta;

   private:
    std::string profile_name = "W8A8B8O8LinearReLU";
};

void load_W8A8B8O8LinearReLU_params(W8A8B8O8LinearReLU &op, std::string prefix);
