#include "common.h"
#include "utils.h"

class Linear_FP {
public:
    Linear_FP(Matrix3D<float> weight_, std::string head_path): weight(weight_) {
        read_to_array((head_path).c_str(), this->weight.m_data, this->weight.length());
    };
    Linear_FP() {};
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    Matrix3D<float> weight;
private:
    std::string profile_name = "Linear_FP";
};
