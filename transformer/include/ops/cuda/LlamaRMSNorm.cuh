#include "common.h"
// #include "common.cuh"

class LlamaRMSNorm_half {
   public:
    LlamaRMSNorm_half(Matrix3D<float> _weight) : weight(_weight){};
    LlamaRMSNorm_half(){};
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    Matrix3D<float> weight;
    float eps = 1e-6;

   private:
    std::string profile_name = "LlamaRMSNorm_half";
};
