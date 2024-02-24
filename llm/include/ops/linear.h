#ifndef LINEAR_H
#define LINEAR_H
#include "common.h"
#include "utils.h"

class Linear_FP {
   public:
    Linear_FP(Matrix3D<float> weight_, std::string weight_path) : weight(weight_) {
        read_to_array((weight_path).c_str(), this->weight.m_data, this->weight.length());
        has_bias = false;
    };
    Linear_FP(Matrix3D<float> weight_, std::string weight_path, Matrix3D<float> bias_, std::string bias_path)
        : weight(weight_), bias(bias_) {
        read_to_array((weight_path).c_str(), this->weight.m_data, this->weight.length());
        read_to_array((bias_path).c_str(), this->bias.m_data, this->bias.length());
        this->has_bias = true;
    };
    Linear_FP(){};
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    Matrix3D<float> weight, bias;
    bool has_bias;

   private:
    std::string profile_name = "Linear_FP";
};

class Linear_FP_int4 {
   public:
    Linear_FP_int4(Matrix3D<uint8_t> weight_, std::string weight_path) : weight(weight_) {
        float *scale_ptr, *zero_point_ptr;
        // float *offset_ptr;
        // length of int8_t weight = elements / 2
        // length of scales/offset = elements / QK = weight / (QK/2)  // TODO: Currently, we don't need offset
        // length of zero_point = 1
        assert((weight.m_dim_z * 2) % (QK) == 0);
        allocate_aligned_memory(scale_ptr, (this->weight.length() * 2 * sizeof(float)) / QK);
        // allocate_aligned_memory(offset_ptr, (this->weight.length() * 2 * sizeof(float)) / QK);
        allocate_aligned_memory(zero_point_ptr, 1 * sizeof(float));

        int x = this->weight.m_dim_x, y = this->weight.m_dim_y, z = (this->weight.m_dim_z * 2) / QK;
        scale = Matrix3D<float>(scale_ptr, x, y, z);
        // offset = Matrix3D<float>(offset_ptr, x, y, z);
        zero_point = Matrix3D<float>(zero_point_ptr, 1, 1, 1);
        weight.load((weight_path + "/weight_int4.bin").c_str());
        // offset.load((weight_path + "/offset_int4.bin").c_str());  // TODO: Currently, we don't need offset
        scale.load((weight_path + "/scaling_factor_int4.bin").c_str());
        zero_point.load((weight_path + "/zero_point_int4.bin").c_str());

#ifdef PACK_QK
        // pack weights and scales together to increase memory efficiency
        allocate_aligned_memory(packed_weights, ((weight.length() * 2) / QK) * sizeof(pack_q4_tensor));
        struct pack_q4_tensor *t = (struct pack_q4_tensor *)packed_weights;
        int num_blocks = (weight.length() * 2) / QK;
        for (int i = 0; i < num_blocks; i++) {
            int weight_idx = i * (QK / 2);
            memcpy(t[i].qx, &weight.m_data[weight_idx], (QK / 2) * sizeof(uint8_t));
            t[i].scale = scale.m_data[i];
        }
        // deallocate
        deallocate_memory(weight.m_data);
        deallocate_memory(scale.m_data);
#endif
    };

    Linear_FP_int4(Matrix3D<uint8_t> weight_, std::string weight_path, Matrix3D<float> bias_, std::string bias_path)
        : weight(weight_), bias(bias_) {
        float *scale_ptr, *zero_point_ptr;
        // float *offset_ptr;
        // length of int8_t weight = elements / 2
        // length of scales/offset = elements / QK = weight / (QK/2)  // TODO: Currently, we don't need offset
        // length of zero_point = 1
        assert((weight.m_dim_z * 2) % (QK) == 0);
        allocate_aligned_memory(scale_ptr, (this->weight.length() * 2 * sizeof(float)) / QK);
        // allocate_aligned_memory(offset_ptr, (this->weight.length() * 2 * sizeof(float)) / QK);
        allocate_aligned_memory(zero_point_ptr, 1 * sizeof(float));

        int x = this->weight.m_dim_x, y = this->weight.m_dim_y, z = (this->weight.m_dim_z * 2) / QK;
        scale = Matrix3D<float>(scale_ptr, x, y, z);
        // offset = Matrix3D<float>(offset_ptr, x, y, z);
        zero_point = Matrix3D<float>(zero_point_ptr, 1, 1, 1);
        weight.load((weight_path + "/weight_int4.bin").c_str());
        // offset.load((weight_path + "/offset_int4.bin").c_str());  // TODO: Currently, we don't need offset
        scale.load((weight_path + "/scaling_factor_int4.bin").c_str());
        zero_point.load((weight_path + "/zero_point_int4.bin").c_str());

        read_to_array((bias_path).c_str(), this->bias.m_data, this->bias.length());
        this->has_bias = true;

#ifdef PACK_QK
        throw("Not supported!");
#endif
    };

    Linear_FP_int4(Matrix3D<uint8_t> weight_, std::string weight_path, Matrix3D<float> scale_ptr_,
                   Matrix3D<float> offset_ptr_, Matrix3D<float> zero_point_ptr_)
        : weight(weight_), scale(scale_ptr_), offset(offset_ptr_), zero_point(zero_point_ptr_) {
        // length of int8_t weight = elements / 2
        // length of scales/offset = elements / QK = weight / (QK/2)
        // length of zero_point = 1
        assert((weight.m_dim_z * 2) % (QK) == 0);
        
        weight.load((weight_path + "/weight_int4.bin").c_str());
        // offset.load((weight_path + "/offset_int4.bin").c_str());  // TODO: Currently, we don't need offset
        scale.load((weight_path + "/scaling_factor_int4.bin").c_str());
        zero_point.load((weight_path + "/zero_point_int4.bin").c_str());

#ifdef PACK_QK
        // pack weights and scales together to increase memory efficiency
        allocate_aligned_memory(packed_weights, ((weight.length() * 2) / QK) * sizeof(pack_q4_tensor));
        struct pack_q4_tensor *t = (struct pack_q4_tensor *)packed_weights;
        int num_blocks = (weight.length() * 2) / QK;
        for (int i = 0; i < num_blocks; i++) {
            int weight_idx = i * (QK / 2);
            memcpy(t[i].qx, &weight.m_data[weight_idx], (QK / 2) * sizeof(uint8_t));
            t[i].scale = scale.m_data[i];
        }
        // deallocate
        deallocate_memory(weight.m_data);
        deallocate_memory(scale.m_data);
#endif
    };
    
    Linear_FP_int4(){};
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
    void forward_ref(const Matrix3D<float> &x, Matrix3D<float> &output);
    void forward_fast(const Matrix3D<float> &x, Matrix3D<float> &output);
#ifdef USE_INT8_INT4_PRODUCT
    static void initialize_memory(const int block_size);
#endif
#ifdef QM_ARM
    static void initialize_weight_memory();
#endif
    Matrix3D<uint8_t> weight;
    Matrix3D<float> scale, zero_point;
    Matrix3D<float> offset;
    Matrix3D<float> bias;
    bool has_bias = false;
#ifdef PACK_QK
    struct pack_q4_tensor *packed_weights;
#endif

   private:
    std::string profile_name = "Linear_FP_int4";
};

#ifdef QM_CUDA
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

class Linear_FP16_int4_ref {
   public:
    Linear_FP16_int4_ref(Matrix3D<int> weight_, std::string weight_path) : weight(weight_) {
        naive_float16_t *scale_ptr;
        // naive_float16_t *offset_ptr;  // TODO: Currently, we don't need offset
        int *zero_point_ptr;
        // length of int8_t weight = elements / 2
        // length of scales/offset = elements / QK = weight / (QK/2)
        // length of zero_point = 1
        // assert((weight.m_dim_z * 8) % (QK) == 0);
        allocate_aligned_memory_gpu(scale_ptr, (this->weight.length() * 8 * sizeof(naive_float16_t)) / QK);
        // allocate_aligned_memory_gpu(offset_ptr, (this->weight.length() * 8 * sizeof(naive_float16_t)) / QK);  //
        // TODO: Currently, we don't need offset
        allocate_aligned_memory_gpu(zero_point_ptr, (this->weight.length() * sizeof(int)) / QK);

        int x = this->weight.m_dim_x, y = this->weight.m_dim_y, z = (this->weight.m_dim_z * 8) / QK;
        scale = Matrix3D<naive_float16_t>(scale_ptr, x, y, z);
        // offset = Matrix3D<naive_float16_t>(offset_ptr, x, y, z);  // TODO: Currently, we don't need offset
        zero_point = Matrix3D<int>(zero_point_ptr, x, y, z / 8);
        weight.load((weight_path + "/weight_int4.bin").c_str());
        // offset.load((weight_path + "/offset_int4.bin").c_str());  // TODO: Currently, we don't need offset
        scale.load((weight_path + "/scaling_factor_int4.bin").c_str());
        zero_point.load((weight_path + "/zero_point_int4.bin").c_str());
    };
    Linear_FP16_int4_ref(){};
    void forward_ref(const Matrix3D<naive_float16_t> &x, Matrix3D<naive_float16_t> &output);
    Matrix3D<int> weight;
    Matrix3D<naive_float16_t> scale;
    Matrix3D<naive_float16_t> offset;  // TODO: Currently, we don't need offset
    Matrix3D<int> zero_point;

   private:
    std::string profile_name = "Linear_FP16_int4_ref";
};

class Linear_half_int4 {
   public:
    Linear_half_int4(Matrix3D<int> weight_, std::string weight_path) : weight(weight_) {
        int output_channel = this->weight.m_dim_y, input_channel = this->weight.m_dim_z * 8;
        
        float16_t *scale_ptr;
        // float16_t *offset_ptr;  // TODO: Currently, we don't need offset
        int *zero_point_ptr;
        // length of int8_t weight = elements / 2
        // length of scales/offset = elements / QK = weight / (QK/2)
        // length of zero_point = 1
        // assert((weight.m_dim_z * 8) % (QK) == 0);
        allocate_aligned_memory_gpu(scale_ptr, output_channel * calculate_zeros_width(input_channel, QK) * 8 * sizeof(float16_t));
        // allocate_aligned_memory(offset_ptr, (this->weight.length() * 8 * sizeof(float16_t)) / QK);  // TODO: Currently, we don't need offset
        // Currently, we don't need offset
        allocate_aligned_memory_gpu(zero_point_ptr, output_channel * calculate_zeros_width(input_channel, QK) * sizeof(int));

        scale = Matrix3D<float16_t>(scale_ptr, 1, output_channel, calculate_zeros_width(input_channel, QK) * 8);
        // offset = Matrix3D<float16_t>(offset_ptr, x, y, z);  // TODO: Currently, we don't need offset
        zero_point = Matrix3D<int>(zero_point_ptr, 1, output_channel, calculate_zeros_width(input_channel, QK));
        weight.load((weight_path + "/weight_int4.bin").c_str());
        // offset.load((weight_path + "/offset_int4.bin").c_str());  // TODO: Currently, we don't need offset
        scale.load((weight_path + "/scaling_factor_int4.bin").c_str());
        zero_point.load((weight_path + "/zero_point_int4.bin").c_str());
    };
    Linear_half_int4(){};
    // void forward(const Matrix3D<float16_t> &x, Matrix3D<float16_t> &output);
    void forward(const Matrix3D<float16_t> &x, Matrix3D<float16_t> &output);
    Matrix3D<int> weight;
    Matrix3D<float16_t> scale;
    Matrix3D<float16_t> offset;  // TODO: Currently, we don't need offset
    Matrix3D<int> zero_point;

   private:
    std::string profile_name = "Linear_half_int4";
};
#endif

#endif
