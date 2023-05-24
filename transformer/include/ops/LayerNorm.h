#include "common.h"

struct LayerNorm_params {
    Matrix3D<float> weight;
    Matrix3D<float> bias;
};

class LayerNorm {
public:
    LayerNorm(LayerNorm_params &params_): params(params_) {};
    LayerNorm(){};
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    struct LayerNorm_params params;
private:
    std::string profile_name = "LayerNorm";
};

void load_LayerNorm(LayerNorm &op, std::string prefix);