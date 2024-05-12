#include "utils.h"
#include "common.h"

class LlamaRMSNorm_metal {
   public:
    LlamaRMSNorm_metal(Matrix3D<float> _weight) : weight(_weight){};
    LlamaRMSNorm_metal(){};
    void forward(const Matrix3D<half> &x, Matrix3D<half> &output, float eps);
    Matrix3D<float> weight;
    // half half_eps = 6.10352e-05;

   private:
    std::string profile_name = "LlamaRMSNorm_metal";
};