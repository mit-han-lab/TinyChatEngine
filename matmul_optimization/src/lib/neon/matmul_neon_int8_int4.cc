#include <arm_neon.h>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"

void quantize_fp32_to_int8(float* A, int8_t* qA, float* sA, int size, int block_size) {
    assert(size % block_size == 0);
    assert(block_size == 32);
    int num_block = size / 32;

    for (int i = 0; i < num_block; i++) {
        float32x4_t srcv[8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        int8_t* start_qA = &qA[i * 32];

        for (int l = 0; l < 8; l++) srcv[l] = vld1q_f32(A + i * 32 + 4 * l);
        for (int l = 0; l < 8; l++) asrcv[l] = vabsq_f32(srcv[l]);

        for (int l = 0; l < 4; l++) amaxv[2 * l] = vmaxq_f32(asrcv[2 * l], asrcv[2 * l + 1]);
        for (int l = 0; l < 2; l++) amaxv[4 * l] = vmaxq_f32(amaxv[4 * l], amaxv[4 * l + 2]);
        for (int l = 0; l < 1; l++) amaxv[8 * l] = vmaxq_f32(amaxv[8 * l], amaxv[8 * l + 4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

        sA[i] = d;

        // low half
        for (int l = 0; l < 4; l++) {
            const float32x4_t v = vmulq_n_f32(srcv[l], id);
            const int32x4_t vi = vcvtnq_s32_f32(v);

            start_qA[4 * l + 0] = vgetq_lane_s32(vi, 0);
            start_qA[4 * l + 1] = vgetq_lane_s32(vi, 1);
            start_qA[4 * l + 2] = vgetq_lane_s32(vi, 2);
            start_qA[4 * l + 3] = vgetq_lane_s32(vi, 3);
        }

        // high half
        for (int l = 4; l < 8; l++) {
            const float32x4_t v = vmulq_n_f32(srcv[l], id);
            const int32x4_t vi = vcvtnq_s32_f32(v);

            start_qA[4 * l + 0] = vgetq_lane_s32(vi, 0);
            start_qA[4 * l + 1] = vgetq_lane_s32(vi, 1);
            start_qA[4 * l + 2] = vgetq_lane_s32(vi, 2);
            start_qA[4 * l + 3] = vgetq_lane_s32(vi, 3);
        }
    }
}

void matmul_int8_int4_no_offset(struct matmul_params* params) {
    int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
    assert(params->block_size == 32);

    const int num_block = k / block_size;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float32x4_t sumv0 = vdupq_n_f32(0.0f);
            float32x4_t sumv1 = vdupq_n_f32(0.0f);
            const unsigned char* w_start = &params->B.int4_data_ptr[j * k / 2];
            const signed char* a_start = &params->A.int8_data_ptr[i * k];
            float* s_a = &params->A_scales[i * k / 32];
            float* s_w = &params->scales[j * k / 32];

            const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
            const int8x16_t offsets = vdupq_n_s8(8);
            for (int i = 0; i < num_block; i += 2) {
                int32x4_t int_sum0 = vdupq_n_s32(0);
                int32x4_t int_sum1 = vdupq_n_s32(0);
                float s_0 = *s_a++ * *s_w++;
                float s_1 = *s_a++ * *s_w++;

                const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
                const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
                w_start += 32;

                // Quantization Method Q4_4, convert 64 4-bit to 64 8-bit
                // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
                // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
                // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
                // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
                int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
                int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));

                // apply offset
                w0_low = vsubq_s8(w0_low, offsets);
                w0_high = vsubq_s8(w0_high, offsets);
                w1_low = vsubq_s8(w1_low, offsets);
                w1_high = vsubq_s8(w1_high, offsets);

                // load 64 8-bit activation
                const int8x16_t a0 = vld1q_s8(a_start);
                const int8x16_t a1 = vld1q_s8(a_start + 16);
                const int8x16_t a2 = vld1q_s8(a_start + 32);
                const int8x16_t a3 = vld1q_s8(a_start + 48);
                a_start += 64;

                // dot product into int32x4_t
                int_sum0 = vdotq_s32(int_sum0, w0_low, a0);
                int_sum0 = vdotq_s32(int_sum0, w0_high, a1);
                int_sum1 = vdotq_s32(int_sum1, w1_low, a2);
                int_sum1 = vdotq_s32(int_sum1, w1_high, a3);

                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
            }
            params->C.data_ptr[i * n + j] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
        }
    }
}

namespace matmul {
void MatmulOperator::mat_mul_accelerator_int8_int4_fast_no_offset(struct matmul_params* params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    assert(params->block_size % 32 == 0);  // support block size to be multiply of 32
    assert(A->row == C->row);              // support block size to be multiply of 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);
    matmul_int8_int4_no_offset(params);
};
}  // namespace matmul
