#include "common.h"
#include <limits>

struct Conv2D_params {
    Matrix4D<float> weight;
    Matrix3D<float> bias;
    int stride_width = 1;
    int stride_height = 1;
    int dilation_width_factor = 1;
    int dilation_height_factor = 1;
    int padding_width = 0;
    int padding_height = 0;
    float float_activation_min = -std::numeric_limits<float>::max();
    float float_activation_max = std::numeric_limits<float>::max();
};

class Conv2D {
   public:
    Conv2D(Conv2D_params params_) : params(params_){};
    Conv2D(){};
    void forward(const Matrix3D<float> &input, Matrix3D<float> &output);
    struct Conv2D_params params;
    bool has_bias = false;

   private:
    std::string profile_name = "Conv2D";
};

void load_Conv2D(Conv2D &op, std::string prefix);
