#include "common.h"
#include "utils.h"

class LlamaRMSNorm {
   public:
    LlamaRMSNorm(Matrix3D<float> _weight) : weight(_weight){};
    LlamaRMSNorm(){};
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output, float eps);
    Matrix3D<float> weight;

   private:
    std::string profile_name = "LlamaRMSNorm";
};
