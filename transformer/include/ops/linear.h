#include "common.h"
#include "utils.h"

class Linear_FP {
   public:
    Linear_FP(Matrix3D<float> weight_, std::string weight_path) : weight(weight_) {
        read_to_array((weight_path).c_str(), this->weight.m_data, this->weight.length());
    };
    Linear_FP(){};
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    Matrix3D<float> weight;

   private:
    std::string profile_name = "Linear_FP";
};

// class Linear_FP_int4 {
//    public:
//     Linear_FP_int4(Matrix3D<uint8_t> weight_, std::string weight_path) : weight(weight_) {
//         float *scale_ptr, *offset_ptr, *zero_point_ptr;
//         // length of int8_t weight = elements / 2
//         // length of scales/offset/zero_point = elements / QK = weight / (QK/2)
//         assert((weight.m_dim_z * 2) % (QK) == 0);
//         allocate_aligned_memory(scale_ptr, (this->weight.length() * 2) / QK);
//         allocate_aligned_memory(offset_ptr, (this->weight.length() * 2) / QK);
//         // allocate_aligned_memory(zero_point_ptr, (this->weight.length() * 2) / QK);

//         int x = this->weight.m_dim_x, y = this->weight.m_dim_y, z = (this->weight.m_dim_z * 2) / QK;
//         scale = Matrix3D<float>(scale_ptr, x, y, z);
//         offset = Matrix3D<float>(offset_ptr, x, y, z);
//         zero_point = Matrix3D<float>(zero_point_ptr, x, y, z);
//         weight.load((weight_path + "/weight.bin").c_str());
//         offset.load((weight_path + "/offset.bin").c_str());
//         scale.load((weight_path + "/scale.bin").c_str());
//         // zero_point.load((weight_path + "/zero_point.bin").c_str());
//     };
//     Linear_FP_int4(){};
//     void forward(const Matrix3D<float> &x, Matrix3D<float> &output) {
//         int i, j, k;
//         assert(output.m_dim_x == 1);
//         assert(output.m_dim_y == x.m_dim_y);
//         assert(output.m_dim_z == weight.m_dim_y);
//         assert(x.m_dim_z / 2 == weight.m_dim_z);
//         for (i = 0; i < output.m_dim_y; i++) {
//             for (j = 0; j < output.m_dim_z; j++) {
//                 float acc = 0;
//                 for (k = 0; k < weight.m_dim_z; k += QK) {
//                     float s = scale(0, j, k / 32);
//                     float o = offset(0, j, k / 32);
//                     uint8_t *weight_32_int4 = &weight.m_data[j * weight.m_dim_z + k / 2];
//                     float *x_ptr = &x.m_data[i * x.m_dim_z + k];
//                     for (int qi = 0; qi < QK / 2; qi++) {
//                         uint8_t packed_int4 = weight_32_int4[qi];
//                         float deq_0 = (float)(packed_int4 & 0x0F) * s + o;
//                         float deq_1 = (float)(packed_int4 >> 4) * s + o;
//                         acc += *x_ptr++ * deq_0;
//                         acc += *x_ptr++ * deq_1;
//                     }
//                 }
//                 output(0, i, j) = acc;
//             }
//         }
//     };
//     void forward_fast(const Matrix3D<float> &x, Matrix3D<float> &output);
//     Matrix3D<uint8_t> weight;
//     Matrix3D<float> scale, offset, zero_point;

//    private:
//     std::string profile_name = "Linear_FP_int4";
// };

class Linear_FP_int4 {
   public:
    Linear_FP_int4(Matrix3D<uint8_t> weight_, std::string weight_path) : weight(weight_) {
        float *scale_ptr, *zero_point_ptr;
        float *offset_ptr;
        // length of int8_t weight = elements / 2
        // length of scales/offset = elements / QK = weight / (QK/2)
        // length of zero_point = 1
        assert((weight.m_dim_z * 2) % (QK) == 0);
        allocate_aligned_memory(scale_ptr, (this->weight.length() * 2 * sizeof(float)) / QK);
        allocate_aligned_memory(offset_ptr, (this->weight.length() * 2 * sizeof(float)) / QK);
        allocate_aligned_memory(zero_point_ptr, 1 * sizeof(float));

        int x = this->weight.m_dim_x, y = this->weight.m_dim_y, z = (this->weight.m_dim_z * 2) / QK;
        scale = Matrix3D<float>(scale_ptr, x, y, z);
        offset = Matrix3D<float>(offset_ptr, x, y, z);
        zero_point = Matrix3D<float>(zero_point_ptr, 1, 1, 1);
        weight.load((weight_path + "/weight_int4.bin").c_str());
        offset.load((weight_path + "/offset_int4.bin").c_str());
        scale.load((weight_path + "/scaling_factor_int4.bin").c_str());
        zero_point.load((weight_path + "/zero_point_int4.bin").c_str());
    };
    Linear_FP_int4(){};
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    void forward_ref(const Matrix3D<float> &x, Matrix3D<float> &output);
    void forward_fast(const Matrix3D<float> &x, Matrix3D<float> &output);
    Matrix3D<uint8_t> weight;
    Matrix3D<float> scale, zero_point;
    Matrix3D<float> offset;

   private:
    std::string profile_name = "Linear_FP_int4";
};
