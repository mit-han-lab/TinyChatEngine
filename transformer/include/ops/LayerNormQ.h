#include "common.h"

struct LayerNormQ_params {
    Matrix3D<float> weight;
    Matrix3D<float> bias;
};

class LayerNormQ {
public:
    LayerNormQ(LayerNormQ_params &params_): params(params_) {};
    LayerNormQ(){};
    void forward(const Matrix3D<float> &x, Matrix3D<int8_t> &output);
    struct LayerNormQ_params params;
private:
    std::string profile_name = "LayerNormQ";
};

void load_LayerNormQ(LayerNormQ &op, std::string prefix);