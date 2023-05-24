#include "common.h"

struct BMM_S8T_S8N_S8T_params {
    float alpha;
};

class BMM_S8T_S8N_S8T{
public:
    BMM_S8T_S8N_S8T(BMM_S8T_S8N_S8T_params &params_);
    BMM_S8T_S8N_S8T(){};
    void forward(const Matrix3D<int8_t> &x, const Matrix3D<int8_t> &weight, Matrix3D<int8_t> &output);
    struct matmul_params params;
    float alpha;
private:
    std::string profile_name = "BMM_S8T_S8N_S8T";
};

void load_BMM_S8T_S8N_S8T(BMM_S8T_S8N_S8T &op, std::string prefix);