#include "common.h"
#include "utils.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

class Linear_FP16_int4_ref {
   public:
    Linear_FP16_int4_ref(Matrix3D<int> weight_, std::string weight_path) : weight(weight_) {
        float16_t *scale_ptr;
        // float16_t *offset_ptr;  // TODO: Currently, we don't need offset
        int *zero_point_ptr;
        // length of int8_t weight = elements / 2
        // length of scales/offset = elements / QK = weight / (QK/2)
        // length of zero_point = 1
        // assert((weight.m_dim_z * 8) % (QK) == 0);
        allocate_aligned_memory_gpu(scale_ptr, (this->weight.length() * 8 * sizeof(float16_t)) / QK);
        // allocate_aligned_memory_gpu(offset_ptr, (this->weight.length() * 8 * sizeof(float16_t)) / QK);  // TODO: Currently, we don't need offset
        allocate_aligned_memory_gpu(zero_point_ptr, (this->weight.length() * sizeof(int)) / QK);

        int x = this->weight.m_dim_x, y = this->weight.m_dim_y, z = (this->weight.m_dim_z * 8) / QK;
        scale = Matrix3D<float16_t>(scale_ptr, x, y, z);
        // offset = Matrix3D<float16_t>(offset_ptr, x, y, z);  // TODO: Currently, we don't need offset
        zero_point = Matrix3D<int>(zero_point_ptr, x, y, z / 8);
        weight.load((weight_path + "/weight_int4.bin").c_str());
        // offset.load((weight_path + "/offset_int4.bin").c_str());  // TODO: Currently, we don't need offset
        scale.load((weight_path + "/scaling_factor_int4.bin").c_str());
        zero_point.load((weight_path + "/zero_point_int4.bin").c_str());
    };
    Linear_FP16_int4_ref(){};
    void forward_ref(const Matrix3D<float16_t> &x, Matrix3D<float16_t> &output);
    Matrix3D<int> weight;
    Matrix3D<float16_t> scale;
    Matrix3D<float16_t> offset;  // TODO: Currently, we don't need offset
    Matrix3D<int> zero_point;

   private:
    std::string profile_name = "Linear_FP16_int4_ref";
};

class Linear_half_int4_ref {
   public:
    Linear_half_int4_ref(Matrix3D<int> weight_, std::string weight_path) : weight(weight_) {
        half *scale_ptr;
        // half *offset_ptr;  // TODO: Currently, we don't need offset
        int *zero_point_ptr;
        // length of int8_t weight = elements / 2
        // length of scales/offset = elements / QK = weight / (QK/2)
        // length of zero_point = 1
        // assert((weight.m_dim_z * 8) % (QK) == 0);
        allocate_aligned_memory_gpu(scale_ptr, (this->weight.length() * 8 * sizeof(half)) / QK);
        // allocate_aligned_memory(offset_ptr, (this->weight.length() * 8 * sizeof(half)) / QK);  // TODO: Currently, we don't need offset
        allocate_aligned_memory_gpu(zero_point_ptr, (this->weight.length() * sizeof(int)) / QK);

        int x = this->weight.m_dim_x, y = this->weight.m_dim_y, z = (this->weight.m_dim_z * 8) / QK;
        scale = Matrix3D<half>(scale_ptr, x, y, z);
        // offset = Matrix3D<half>(offset_ptr, x, y, z);  // TODO: Currently, we don't need offset
        zero_point = Matrix3D<int>(zero_point_ptr, x, y, z / 8);
        weight.load((weight_path + "/weight_int4.bin").c_str());
        // offset.load((weight_path + "/offset_int4.bin").c_str());  // TODO: Currently, we don't need offset
        scale.load((weight_path + "/scaling_factor_int4.bin").c_str());
        zero_point.load((weight_path + "/zero_point_int4.bin").c_str());
    };
    Linear_half_int4_ref(){};
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    Matrix3D<int> weight;
    Matrix3D<half> scale;
    Matrix3D<half> offset;  // TODO: Currently, we don't need offset
    Matrix3D<int> zero_point;

   private:
    std::string profile_name = "Linear_half_int4_ref";
};

class Linear_half_int4 {
   public:
    Linear_half_int4(Matrix3D<int> weight_, std::string weight_path) : weight(weight_) {
        half *scale_ptr;
        // half *offset_ptr;  // TODO: Currently, we don't need offset
        int *zero_point_ptr;
        // length of int8_t weight = elements / 2
        // length of scales/offset = elements / QK = weight / (QK/2)
        // length of zero_point = 1
        // assert((weight.m_dim_z * 8) % (QK) == 0);
        allocate_aligned_memory_gpu(scale_ptr, (this->weight.length() * 8 * sizeof(half)) / QK);
        // allocate_aligned_memory(offset_ptr, (this->weight.length() * 8 * sizeof(half)) / QK);  // TODO: Currently, we don't need offset
        allocate_aligned_memory_gpu(zero_point_ptr, (this->weight.length() * sizeof(int)) / QK);

        int x = this->weight.m_dim_x, y = this->weight.m_dim_y, z = (this->weight.m_dim_z * 8) / QK;
        scale = Matrix3D<half>(scale_ptr, x, y, z);
        // offset = Matrix3D<half>(offset_ptr, x, y, z);  // TODO: Currently, we don't need offset
        zero_point = Matrix3D<int>(zero_point_ptr, x, y, z / 8);
        weight.load((weight_path + "/weight_int4.bin").c_str());
        // offset.load((weight_path + "/offset_int4.bin").c_str());  // TODO: Currently, we don't need offset
        scale.load((weight_path + "/scaling_factor_int4.bin").c_str());
        zero_point.load((weight_path + "/zero_point_int4.bin").c_str());
    };
    Linear_half_int4(){};
    void forward(const Matrix3D<half> &x, Matrix3D<half> &output);
    Matrix3D<int> weight;
    Matrix3D<half> scale;
    Matrix3D<half> offset;  // TODO: Currently, we don't need offset
    Matrix3D<int> zero_point;

   private:
    std::string profile_name = "Linear_half_int4";
};

class Linear_half_int4_test {
   public:
    Linear_half_int4_test(Matrix3D<int> weight_, std::string weight_path) : weight(weight_) {
        half *scale_ptr;
        // half *offset_ptr;  // TODO: Currently, we don't need offset
        int *zero_point_ptr;
        // length of int8_t weight = elements / 2
        // length of scales/offset = elements / QK = weight / (QK/2)
        // length of zero_point = 1
        // assert((weight.m_dim_z * 8) % (QK) == 0);
        allocate_aligned_memory_gpu(scale_ptr, (this->weight.length() * 8 * sizeof(half)) / QK);
        // allocate_aligned_memory(offset_ptr, (this->weight.length() * 8 * sizeof(half)) / QK);  // TODO: Currently, we don't need offset
        allocate_aligned_memory_gpu(zero_point_ptr, (this->weight.length() * sizeof(int)) / QK);

        int x = this->weight.m_dim_x, y = this->weight.m_dim_y, z = (this->weight.m_dim_z * 8) / QK;
        scale = Matrix3D<half>(scale_ptr, x, y, z);
        // offset = Matrix3D<half>(offset_ptr, x, y, z);  // TODO: Currently, we don't need offset
        zero_point = Matrix3D<int>(zero_point_ptr, x, y, z / 8);
        weight.load((weight_path + "/weight_int4.bin").c_str());
        // offset.load((weight_path + "/offset_int4.bin").c_str());  // TODO: Currently, we don't need offset
        scale.load((weight_path + "/scaling_factor_int4.bin").c_str());
        zero_point.load((weight_path + "/zero_point_int4.bin").c_str());
    };
    Linear_half_int4_test(){};
    void forward(const Matrix3D<half> &x, Matrix3D<half> &output);
    Matrix3D<int> weight;
    Matrix3D<half> scale;
    Matrix3D<half> offset;  // TODO: Currently, we don't need offset
    Matrix3D<int> zero_point;

   private:
    std::string profile_name = "Linear_half_int4_test";
};
