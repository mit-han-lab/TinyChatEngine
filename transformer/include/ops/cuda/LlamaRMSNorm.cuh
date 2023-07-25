#include "common.h"

class LlamaRMSNorm_cuda {
   public:
    LlamaRMSNorm_cuda(Matrix3D<float> _weight) : weight(_weight){};
    LlamaRMSNorm_cuda(){};
    void forward(const Matrix3D<half> &x, Matrix3D<half> &output);
    Matrix3D<float> weight;
    float eps = 1e-6;
    // half half_eps = 6.10352e-05;

   private:
    std::string profile_name = "LlamaRMSNorm_cuda";
};
