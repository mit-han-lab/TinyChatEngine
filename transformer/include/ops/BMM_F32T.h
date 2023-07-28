#include "common.h"

class BMM_F32T {
   public:
    BMM_F32T(float _alpha);
    BMM_F32T(){};
    void forward(const Matrix3D<float> &x, const Matrix3D<float> &weight, Matrix3D<float> &output);
    void forward_weight_untransposed(const Matrix3D<float> &x, const Matrix3D<float> &weight, Matrix3D<float> &output);
    float alpha;

   private:
    std::string profile_name = "BMM_F32T";
};

void load_BMM_F32T(BMM_F32T &op, std::string prefix);
