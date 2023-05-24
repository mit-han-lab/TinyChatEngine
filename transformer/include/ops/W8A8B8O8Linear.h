#include "common.h"

struct W8A8B8O8Linear_params {
    Matrix3D<int8_t> weight;
    Matrix3D<int8_t> bias;
    float alpha;
    float beta;
};

class W8A8B8O8Linear {
   public:
    W8A8B8O8Linear(W8A8B8O8Linear_params &params_);
    W8A8B8O8Linear(){};
    void forward(const Matrix3D<int8_t> &x, Matrix3D<int8_t> &output);
    struct matmul_params params;
    float alpha;
    float beta;

   private:
    std::string profile_name = "W8A8B8O8Linear";
};

void load_W8A8B8O8Linear_params(W8A8B8O8Linear &op, std::string prefix);
