#include "common.cuh"

class LlamaRMSNorm_half {
   public:
    LlamaRMSNorm_half(Matrix3D_cuda<float> _weight) : weight(_weight){};
    LlamaRMSNorm_half(){};
    void forward(const Matrix3D_cuda<float> &x, Matrix3D_cuda<float> &output);
    Matrix3D_cuda<float> weight;
    float eps = 1e-6;

   private:
    std::string profile_name = "LlamaRMSNorm_half";
};
