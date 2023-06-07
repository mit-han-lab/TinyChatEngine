#include "common.h"
#include "utils.h"

class Linear_FP {
public:
  Linear_FP(Matrix3D<float> weight_, std::string weight_path)
      : weight(weight_) {
    read_to_array((weight_path).c_str(), this->weight.m_data,
                  this->weight.length());
  };
  Linear_FP(){};
  void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
  Matrix3D<float> weight;

private:
  std::string profile_name = "Linear_FP";
};

class Linear_FP_int4 {
public:
  Linear_FP_int4(Matrix3D<uint8_t> weight_, std::string weight_path)
      : weight(weight_) {
    read_to_array((weight_path).c_str(), this->weight.m_data,
                  this->weight.length());
  };
  Linear_FP_int4(){};
  void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
  Matrix3D<uint8_t> weight;

private:
  std::string profile_name = "Linear_FP_int4";
};
