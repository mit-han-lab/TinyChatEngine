#include "common.h"

struct BMM_S8T_S8N_F32T_params {
    float alpha;
};

class BMM_S8T_S8N_F32T{
public:
    BMM_S8T_S8N_F32T(BMM_S8T_S8N_F32T_params &params_);
    BMM_S8T_S8N_F32T(){};
    void forward(const Matrix3D<int8_t> &x, const Matrix3D<int8_t> &weight, Matrix3D<float> &output);
    struct matmul_params params;
    float alpha;
private:
    std::string profile_name = "BMM_S8T_S8N_F32T";
};

void load_BMM_S8T_S8N_F32T(BMM_S8T_S8N_F32T &op, std::string prefix);