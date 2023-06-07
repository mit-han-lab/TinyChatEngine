#include <immintrin.h>

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

static inline __m256i bytes_from_nibbles_32(const uint8_t *rsi) {
    // Load 16 bytes from memory
    __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);

    // Expand bytes into uint16_t values
    __m256i bytes = _mm256_cvtepu8_epi16(tmp);

    // Unpack values into individual bytes
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    __m256i high = _mm256_andnot_si256(lowMask, bytes);
    __m256i low = _mm256_and_si256(lowMask, bytes);
    high = _mm256_slli_epi16(high, 4);
    bytes = _mm256_or_si256(low, high);
    return bytes;
}

// Dequantize a block of weight
static void dequantize_block_q4_1(const uint8_t *int4_w, float *y, float scale, float offset) {
    const __m256 d_v = _mm256_broadcast_ss(&scale);
    const __m256 d_m = _mm256_broadcast_ss(&offset);

    const uint8_t *pp = int4_w;

    for (int l = 0; l < QK; l += 32) {
        // Load 32x4-bit integers into 32x8-bit integers
        __m256i vx8 = bytes_from_nibbles_32(pp + l / 2);

        // Convert to 16-bit int
        const __m256i vx16_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 0));
        const __m256i vx16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx8, 1));

        // Convert to 32-bit int -> float 32
        const __m256 vf[4] = {_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 0))),
                              _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_lo, 1))),
                              _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 0))),
                              _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(vx16_hi, 1)))};

        // Scale, add m and store
        for (int j = 0; j < 4; j++) {
            const __m256 result = _mm256_add_ps(_mm256_mul_ps(vf[j], d_v), d_m);
            _mm256_storeu_ps(y + l + j * 8, result);
        }
    }
}

struct linear_thread_args {
    int start_j, end_j;
    Matrix3D<uint8_t> weight;
    Matrix3D<float> scale, offset, x, output;
};

static void *fast_over_column_func_v2(void *args) {
    int i, j, k;
    auto *mat_args = (struct linear_thread_args *)args;
    Matrix3D<uint8_t> weight = mat_args->weight;
    Matrix3D<float> scale = mat_args->scale;
    Matrix3D<float> offset = mat_args->offset;
    Matrix3D<float> x = mat_args->x;
    Matrix3D<float> output = mat_args->output;

    float weight_block[QK];
    float weight_block2[QK];

    for (i = 0; i < output.m_dim_y; i++) {
        for (j = mat_args->start_j; j < mat_args->end_j; j += 2) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (k = 0; k < weight.m_dim_z; k += QK) {
                float s = scale(0, j, k / 32), s1 = scale(0, j + 1, k / 32);
                float o = offset(0, j, k / 32), o1 = offset(0, j + 1, k / 32);
                // float zp = zero_point(0, j, k/32);
                uint8_t *weight_32_int4 = &weight.m_data[j * weight.m_dim_z + k / 2];
                uint8_t *weight_32_int4_2 = &weight.m_data[(j + 1) * weight.m_dim_z + k / 2];
                __m256 *x_ptr = (__m256 *)&x.m_data[i * x.m_dim_z + k];
                __m256 *w_ptr = (__m256 *)&weight_block;
                __m256 *w2_ptr = (__m256 *)&weight_block2;
                dequantize_block_q4_1(weight_32_int4, weight_block, s, o);
                dequantize_block_q4_1(weight_32_int4_2, weight_block2, s1, o1);

                // assume QK == 32 (8 x 32 float)
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(*x_ptr, *w_ptr++));
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(*x_ptr++, *w2_ptr++));
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(*x_ptr, *w_ptr++));
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(*x_ptr++, *w2_ptr++));
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(*x_ptr, *w_ptr++));
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(*x_ptr++, *w2_ptr++));
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(*x_ptr, *w_ptr++));
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(*x_ptr++, *w2_ptr++));
            }
            float *ptr = (float *)&acc0;
            output(0, i, j) = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
            ptr = (float *)&acc1;
            output(0, i, j + 1) = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
        }
    }
    return NULL;
}

static void *fast_over_column_func_v1(void *args) {
    int i, j, k;
    auto *mat_args = (struct linear_thread_args *)args;
    Matrix3D<uint8_t> weight = mat_args->weight;
    Matrix3D<float> scale = mat_args->scale;
    Matrix3D<float> offset = mat_args->offset;
    Matrix3D<float> x = mat_args->x;
    Matrix3D<float> output = mat_args->output;

    float weight_block[QK];
    float weight_block2[QK];

    for (i = 0; i < output.m_dim_y; i++) {
        for (j = mat_args->start_j; j < mat_args->end_j; j++) {
            __m256 acc0 = _mm256_setzero_ps();
            for (k = 0; k < weight.m_dim_z; k += QK) {
                float s = scale(0, j, k / 32);
                float o = offset(0, j, k / 32);
                // float zp = zero_point(0, j, k/32);
                uint8_t *weight_32_int4 = &weight.m_data[j * weight.m_dim_z + k / 2];
                __m256 *x_ptr = (__m256 *)&x.m_data[i * x.m_dim_z + k];
                __m256 *w_ptr = (__m256 *)&weight_block;
                dequantize_block_q4_1(weight_32_int4, weight_block, s, o);

                // assume QK == 32 (8 x 32 float)
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(*x_ptr++, *w_ptr++));
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(*x_ptr++, *w_ptr++));
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(*x_ptr++, *w_ptr++));
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(*x_ptr++, *w_ptr++));
            }
            float *ptr = (float *)&acc0;
            output(0, i, j) = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
        }
    }
    return NULL;
}

class Linear_FP_int4 {
   public:
    Linear_FP_int4(Matrix3D<uint8_t> weight_, std::string weight_path) : weight(weight_) {
        float *scale_ptr, *offset_ptr, *zero_point_ptr;
        // length of int8_t weight = elements / 2
        // length of scales/offset/zero_point = elements / QK = weight / (QK/2)
        assert((weight.m_dim_z * 2) % (QK) == 0);
        allocate_aligned_memory(scale_ptr, (this->weight.length() * 2) / QK);
        allocate_aligned_memory(offset_ptr, (this->weight.length() * 2) / QK);
        // allocate_aligned_memory(zero_point_ptr, (this->weight.length() * 2) / QK);

        int x = this->weight.m_dim_x, y = this->weight.m_dim_y, z = (this->weight.m_dim_z * 2) / QK;
        scale = Matrix3D<float>(scale_ptr, x, y, z);
        offset = Matrix3D<float>(offset_ptr, x, y, z);
        zero_point = Matrix3D<float>(zero_point_ptr, x, y, z);
        weight.load((weight_path + "/weight.bin").c_str());
        offset.load((weight_path + "/offset.bin").c_str());
        scale.load((weight_path + "/scale.bin").c_str());
        // zero_point.load((weight_path + "/zero_point.bin").c_str());
    };
    Linear_FP_int4(){};
    void forward(const Matrix3D<float> &x, Matrix3D<float> &output) {
        int i, j, k;
        assert(output.m_dim_x == 1);
        assert(output.m_dim_y == x.m_dim_y);
        assert(output.m_dim_z == weight.m_dim_y);
        assert(x.m_dim_z / 2 == weight.m_dim_z);
        for (i = 0; i < output.m_dim_y; i++) {
            for (j = 0; j < output.m_dim_z; j++) {
                float acc = 0;
                for (k = 0; k < weight.m_dim_z; k += QK) {
                    float s = scale(0, j, k / 32);
                    float o = offset(0, j, k / 32);
                    uint8_t *weight_32_int4 = &weight.m_data[j * weight.m_dim_z + k / 2];
                    float *x_ptr = &x.m_data[i * x.m_dim_z + k];
                    for (int qi = 0; qi < QK / 2; qi++) {
                        uint8_t packed_int4 = weight_32_int4[qi];
                        float deq_0 = (float)(packed_int4 & 0x0F) * s + o;
                        float deq_1 = (float)(packed_int4 >> 4) * s + o;
                        acc += *x_ptr++ * deq_0;
                        acc += *x_ptr++ * deq_1;
                    }
                }
                output(0, i, j) = acc;
            }
        }
    };
    void forward_my(const Matrix3D<float> &x, Matrix3D<float> &output) {
        int i, j, k;
        assert(output.m_dim_x == 1);
        assert(output.m_dim_y == x.m_dim_y);
        assert(output.m_dim_z == weight.m_dim_y);
        assert(x.m_dim_z / 2 == weight.m_dim_z);

        float weight_block[QK];

        for (i = 0; i < output.m_dim_y; i++) {
            for (j = 0; j < output.m_dim_z; j++) {
                float acc = 0;
                for (k = 0; k < weight.m_dim_z; k += QK) {
                    float s = scale(0, j, k / 32);
                    float o = offset(0, j, k / 32);
                    uint8_t *weight_32_int4 = &weight.m_data[j * weight.m_dim_z + k / 2];
                    float *x_ptr = &x.m_data[i * x.m_dim_z + k];
                    dequantize_block_q4_1(weight_32_int4, weight_block, s, o);

                    for (int qi = 0; qi < QK; qi++) {
                        acc += *x_ptr++ * weight_block[qi];
                    }
                }
                output(0, i, j) = acc;
            }
        }
    };

    void forward_fast(const Matrix3D<float> &x, Matrix3D<float> &output) {
        const int num_thread = 8;
        int i, j, k;
        assert(output.m_dim_x == 1);
        assert(output.m_dim_y == x.m_dim_y);
        assert(output.m_dim_z == weight.m_dim_y);
        assert(x.m_dim_z / 2 == weight.m_dim_z);

        assert(output.m_dim_z > num_thread);
        assert(output.m_dim_z % (num_thread * 2) == 0);  // unroll column by 2
        pthread_t thread_pool[num_thread];
        struct linear_thread_args threads_args[num_thread];

        // Thread creation
        for (j = 0; j < num_thread; j++) {
            threads_args[j].start_j = j * (output.m_dim_z / num_thread);
            threads_args[j].end_j = (j + 1) * (output.m_dim_z / num_thread);
            threads_args[j].weight = weight;
            threads_args[j].scale = scale;
            threads_args[j].offset = offset;
            threads_args[j].x = x;
            threads_args[j].output = output;
            pthread_create(&thread_pool[j], NULL, fast_over_column_func_v2, &threads_args[j]);
        }
        // Join threads
        for (j = 0; j < num_thread; j++) {
            pthread_join(thread_pool[j], NULL);
        }
    };
    Matrix3D<uint8_t> weight;
    Matrix3D<float> scale, offset, zero_point;

   private:
    std::string profile_name = "Linear_FP_int4";
};

class Linear_FP_int4 {
public:
  Linear_FP_int4(Matrix3D<uint8_t> weight_, std::string weight_path)
      : weight(weight_) {
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
    //offset = Matrix3D<float>(offset_ptr, x, y, z);
    zero_point = Matrix3D<float>(zero_point_ptr, 1, 1, 1);
    weight.load((weight_path + "/weight_int4.bin").c_str());
    //offset.load((weight_path + "/offset_int4.bin").c_str());
    scale.load((weight_path + "/scaling_factor_int4.bin").c_str());
    zero_point.load((weight_path + "/zero_point_int4.bin").c_str());
  };
  Linear_FP_int4(){};
  void forward(const Matrix3D<float> &x, Matrix3D<float> &output);
  Matrix3D<uint8_t> weight;
  Matrix3D<float> scale, zero_point;
  //Matrix3D<float> offset;

private:
  std::string profile_name = "Linear_FP_int4";
};
