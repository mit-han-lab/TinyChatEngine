#include <arm_neon.h>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <arm_neon.h>

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

#include "../matmul.h"
#include "common.h"
#include "pthread_pool.h"

struct a8w4_thread_args {
    int start_i, end_i, start_j, end_j, tile_size;
    const struct matmul_params* params;
};

// Most of this function is from llama.cpp
void quantize_fp32_to_int8(float* A, int8_t* qA, float* sA, int size, int block_size) {
    assert(size % block_size == 0);
    assert(block_size == 32);
    int num_block = size / 32;

    for (int i = 0; i < num_block; i++) {
        float32x4_t srcv[8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

#ifdef PACK_QK
        struct pack_q8_tensor* t = (struct pack_q8_tensor*)qA;
        int8_t* start_qA = t[i].qx;
#else
        int8_t* start_qA = &qA[i * 32];
#endif

        for (int l = 0; l < 8; l++) srcv[l] = vld1q_f32(A + i * 32 + 4 * l);
        for (int l = 0; l < 8; l++) asrcv[l] = vabsq_f32(srcv[l]);

        for (int l = 0; l < 4; l++) amaxv[2 * l] = vmaxq_f32(asrcv[2 * l], asrcv[2 * l + 1]);
        for (int l = 0; l < 2; l++) amaxv[4 * l] = vmaxq_f32(amaxv[4 * l], amaxv[4 * l + 2]);
        for (int l = 0; l < 1; l++) amaxv[8 * l] = vmaxq_f32(amaxv[8 * l], amaxv[8 * l + 4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;

#ifdef PACK_QK
        t[i].scale = d;
#else
        sA[i] = d;
#endif

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

void dequantize_int4_to_fp32(uint8_t* qW, float* W, float* sW, int size, int block_size) {
    assert(size % block_size == 0);
    assert(block_size == 32);
    int num_block = size / 32;

    const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
    const int8x16_t offsets = vdupq_n_s8(8);
    float* w_start_fp32 = &W[0];
    for (int i = 0; i < num_block; i++) {
        const unsigned char* w_start = &qW[i * 16];
        float* s_w = &sW[i];
        float s_0 = s_w[0];

        const uint8x16_t w0 = vld1q_u8(w_start);  // 32 4bit weight

        // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
        // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
        // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
        // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
        // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
        int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
        int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));

        // apply offset
        w0_low = vsubq_s8(w0_low, offsets);
        w0_high = vsubq_s8(w0_high, offsets);

        // Step 1: Split each int8x16_t vector into two int8x8_t vectors
        int8x8_t w0_low_low = vget_low_s8(w0_low);
        int8x8_t w0_low_high = vget_high_s8(w0_low);
        int8x8_t w0_high_low = vget_low_s8(w0_high);
        int8x8_t w0_high_high = vget_high_s8(w0_high);

        // Step 2: Extend each int8x8_t vector to int16x8_t
        int16x8_t w0_ll_ext = vmovl_s8(w0_low_low);
        int16x8_t w0_lh_ext = vmovl_s8(w0_low_high);
        int16x8_t w0_hl_ext = vmovl_s8(w0_high_low);
        int16x8_t w0_hh_ext = vmovl_s8(w0_high_high);

        // Step 3: Further extend int16x8_t to int32x4_t and then convert to float32x4_t
        float32x4_t w0_ll_f = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0_ll_ext)));
        float32x4_t w0_lh_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0_ll_ext)));
        float32x4_t w0_hl_f = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0_lh_ext)));
        float32x4_t w0_hh_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0_lh_ext)));
        float32x4_t w1_ll_f = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0_hl_ext)));
        float32x4_t w1_lh_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0_hl_ext)));
        float32x4_t w1_hl_f = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0_hh_ext)));
        float32x4_t w1_hh_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0_hh_ext)));

        float32x4_t sumv0_ll = vmulq_n_f32(w0_ll_f, s_0);
        float32x4_t sumv0_lh = vmulq_n_f32(w0_lh_f, s_0);
        float32x4_t sumv0_hl = vmulq_n_f32(w0_hl_f, s_0);
        float32x4_t sumv0_hh = vmulq_n_f32(w0_hh_f, s_0);
        float32x4_t sumv1_ll = vmulq_n_f32(w1_ll_f, s_0);
        float32x4_t sumv1_lh = vmulq_n_f32(w1_lh_f, s_0);
        float32x4_t sumv1_hl = vmulq_n_f32(w1_hl_f, s_0);
        float32x4_t sumv1_hh = vmulq_n_f32(w1_hh_f, s_0);

        vst1q_f32(w_start_fp32, sumv0_ll);
        w_start_fp32 += 4;
        vst1q_f32(w_start_fp32, sumv0_lh);
        w_start_fp32 += 4;
        vst1q_f32(w_start_fp32, sumv0_hl);
        w_start_fp32 += 4;
        vst1q_f32(w_start_fp32, sumv0_hh);
        w_start_fp32 += 4;
        vst1q_f32(w_start_fp32, sumv1_ll);
        w_start_fp32 += 4;
        vst1q_f32(w_start_fp32, sumv1_lh);
        w_start_fp32 += 4;
        vst1q_f32(w_start_fp32, sumv1_hl);
        w_start_fp32 += 4;
        vst1q_f32(w_start_fp32, sumv1_hh);
        w_start_fp32 += 4;
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
            float32x4_t sumv2 = vdupq_n_f32(0.0f);
            float32x4_t sumv3 = vdupq_n_f32(0.0f);
            const unsigned char* w_start = &params->B.int4_data_ptr[j * k / 2];
            const signed char* a_start = &params->A.int8_data_ptr[i * k];
            float* s_a = &params->A_scales[i * k / 32];
            float* s_w = &params->scales[j * k / 32];

            const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
            const int8x16_t offsets = vdupq_n_s8(8);
            for (int q = 0; q < num_block; q += 4) {
                int32x4_t int_sum0 = vdupq_n_s32(0);
                int32x4_t int_sum1 = vdupq_n_s32(0);
                int32x4_t int_sum2 = vdupq_n_s32(0);
                int32x4_t int_sum3 = vdupq_n_s32(0);
                float s_0 = *s_a++ * *s_w++;
                float s_1 = *s_a++ * *s_w++;
                float s_2 = *s_a++ * *s_w++;
                float s_3 = *s_a++ * *s_w++;

                const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
                const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
                const uint8x16_t w2 = vld1q_u8(w_start + 32);  // 32 4bit weight
                const uint8x16_t w3 = vld1q_u8(w_start + 48);  // 32 4bit weight
                w_start += 64;

                // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
                // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
                // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
                // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
                // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
                int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
                int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
                int8x16_t w2_low = vreinterpretq_s8_u8(vandq_u8(w2, mask_low4bit));
                int8x16_t w2_high = vreinterpretq_s8_u8(vshrq_n_u8(w2, 4));
                int8x16_t w3_low = vreinterpretq_s8_u8(vandq_u8(w3, mask_low4bit));
                int8x16_t w3_high = vreinterpretq_s8_u8(vshrq_n_u8(w3, 4));

                // apply offset
                w0_low = vsubq_s8(w0_low, offsets);
                w0_high = vsubq_s8(w0_high, offsets);
                w1_low = vsubq_s8(w1_low, offsets);
                w1_high = vsubq_s8(w1_high, offsets);
                w2_low = vsubq_s8(w2_low, offsets);
                w2_high = vsubq_s8(w2_high, offsets);
                w3_low = vsubq_s8(w3_low, offsets);
                w3_high = vsubq_s8(w3_high, offsets);

                // load 64 8-bit activation
                const int8x16_t a0 = vld1q_s8(a_start);
                const int8x16_t a1 = vld1q_s8(a_start + 16);
                const int8x16_t a2 = vld1q_s8(a_start + 32);
                const int8x16_t a3 = vld1q_s8(a_start + 48);
                const int8x16_t a4 = vld1q_s8(a_start + 64);
                const int8x16_t a5 = vld1q_s8(a_start + 80);
                const int8x16_t a6 = vld1q_s8(a_start + 96);
                const int8x16_t a7 = vld1q_s8(a_start + 112);
                a_start += 128;

                // dot product into int32x4_t
                int_sum0 = my_vdotq_s32(int_sum0, w0_low, a0);
                int_sum0 = my_vdotq_s32(int_sum0, w0_high, a1);
                int_sum1 = my_vdotq_s32(int_sum1, w1_low, a2);
                int_sum1 = my_vdotq_s32(int_sum1, w1_high, a3);
                int_sum2 = my_vdotq_s32(int_sum2, w2_low, a4);
                int_sum2 = my_vdotq_s32(int_sum2, w2_high, a5);
                int_sum3 = my_vdotq_s32(int_sum3, w3_low, a6);
                int_sum3 = my_vdotq_s32(int_sum3, w3_high, a7);

                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
                sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(int_sum2), s_2);
                sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(int_sum3), s_3);
            }
            params->C.data_ptr[i * n + j] =
                vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
        }
    }
}

static void* matmul_int8_int4_no_offset_over_column(void* args) {
    struct a8w4_thread_args* mat_args = (struct a8w4_thread_args*)args;
    const struct matmul_params* params = mat_args->params;
    int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
    const int num_block = k / block_size;

    for (int i = 0; i < m; i++) {
        for (int j = mat_args->start_j; j < mat_args->end_j; j++) {
            float32x4_t sumv0 = vdupq_n_f32(0.0f);
            float32x4_t sumv1 = vdupq_n_f32(0.0f);
            const unsigned char* w_start = &params->B.int4_data_ptr[j * k / 2];
            const signed char* a_start = &params->A.int8_data_ptr[i * k];
            float* s_a = &params->A_scales[i * k / 32];
            float* s_w = &params->scales[j * k / 32];

            const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
            const int8x16_t offsets = vdupq_n_s8(8);
            for (int q = 0; q < num_block; q += 2) {
                int32x4_t int_sum0 = vdupq_n_s32(0);
                int32x4_t int_sum1 = vdupq_n_s32(0);
                float s_0 = *s_a++ * *s_w++;
                float s_1 = *s_a++ * *s_w++;

                const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
                const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
                w_start += 32;

                // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
                // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
                // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
                // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
                // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
                int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
                int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));

                // load 64 8-bit activation
                const int8x16_t a0 = vld1q_s8(a_start);
                const int8x16_t a1 = vld1q_s8(a_start + 16);
                const int8x16_t a2 = vld1q_s8(a_start + 32);
                const int8x16_t a3 = vld1q_s8(a_start + 48);
                a_start += 64;

                // apply offset
                w0_low = vsubq_s8(w0_low, offsets);
                w0_high = vsubq_s8(w0_high, offsets);
                w1_low = vsubq_s8(w1_low, offsets);
                w1_high = vsubq_s8(w1_high, offsets);

                // dot product into int32x4_t
                int_sum0 = my_vdotq_s32(int_sum0, w0_low, a0);
                int_sum1 = my_vdotq_s32(int_sum1, w1_low, a2);
                int_sum0 = my_vdotq_s32(int_sum0, w0_high, a1);
                int_sum1 = my_vdotq_s32(int_sum1, w1_high, a3);

                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
            }
            params->C.data_ptr[i * n + j] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
        }
    }

    return NULL;
}

inline static void* gemv_int8_int4_no_offset_over_column_unroll128(void* args) {
    struct a8w4_thread_args* mat_args = (struct a8w4_thread_args*)args;
    const struct matmul_params* params = mat_args->params;
    int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
    const int num_block = k / block_size;
    assert(m == 1);

    for (int j = mat_args->start_j; j < mat_args->end_j; j++) {
        float32x4_t sumv0 = vdupq_n_f32(0.0f);
        float32x4_t sumv1 = vdupq_n_f32(0.0f);
        float32x4_t sumv2 = vdupq_n_f32(0.0f);
        float32x4_t sumv3 = vdupq_n_f32(0.0f);
        const unsigned char* w_start = &params->B.int4_data_ptr[j * k / 2];
        const signed char* a_start = &params->A.int8_data_ptr[0];
        float* s_a = &params->A_scales[0];
        float* s_w = &params->scales[j * k / 32];

        const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
        const int8x16_t offsets = vdupq_n_s8(8);
        for (int q = 0; q < num_block; q += 4) {
            int32x4_t int_sum0 = vdupq_n_s32(0);
            int32x4_t int_sum1 = vdupq_n_s32(0);
            int32x4_t int_sum2 = vdupq_n_s32(0);
            int32x4_t int_sum3 = vdupq_n_s32(0);
            float s_0 = *s_a++ * *s_w++;
            float s_1 = *s_a++ * *s_w++;
            float s_2 = *s_a++ * *s_w++;
            float s_3 = *s_a++ * *s_w++;

            const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
            const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
            const uint8x16_t w2 = vld1q_u8(w_start + 32);  // 32 4bit weight
            const uint8x16_t w3 = vld1q_u8(w_start + 48);  // 32 4bit weight
            w_start += 64;

            // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
            // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
            // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
            // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
            // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
            int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
            int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
            int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
            int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
            int8x16_t w2_low = vreinterpretq_s8_u8(vandq_u8(w2, mask_low4bit));
            int8x16_t w2_high = vreinterpretq_s8_u8(vshrq_n_u8(w2, 4));
            int8x16_t w3_low = vreinterpretq_s8_u8(vandq_u8(w3, mask_low4bit));
            int8x16_t w3_high = vreinterpretq_s8_u8(vshrq_n_u8(w3, 4));

            // apply offset
            w0_low = vsubq_s8(w0_low, offsets);
            w0_high = vsubq_s8(w0_high, offsets);
            w1_low = vsubq_s8(w1_low, offsets);
            w1_high = vsubq_s8(w1_high, offsets);
            w2_low = vsubq_s8(w2_low, offsets);
            w2_high = vsubq_s8(w2_high, offsets);
            w3_low = vsubq_s8(w3_low, offsets);
            w3_high = vsubq_s8(w3_high, offsets);

            // load 64 8-bit activation
            const int8x16_t a0 = vld1q_s8(a_start);
            const int8x16_t a1 = vld1q_s8(a_start + 16);
            const int8x16_t a2 = vld1q_s8(a_start + 32);
            const int8x16_t a3 = vld1q_s8(a_start + 48);
            const int8x16_t a4 = vld1q_s8(a_start + 64);
            const int8x16_t a5 = vld1q_s8(a_start + 80);
            const int8x16_t a6 = vld1q_s8(a_start + 96);
            const int8x16_t a7 = vld1q_s8(a_start + 112);
            a_start += 128;

            // dot product into int32x4_t
            int_sum0 = my_vdotq_s32(int_sum0, w0_low, a0);
            int_sum0 = my_vdotq_s32(int_sum0, w0_high, a1);
            int_sum1 = my_vdotq_s32(int_sum1, w1_low, a2);
            int_sum1 = my_vdotq_s32(int_sum1, w1_high, a3);
            int_sum2 = my_vdotq_s32(int_sum2, w2_low, a4);
            int_sum2 = my_vdotq_s32(int_sum2, w2_high, a5);
            int_sum3 = my_vdotq_s32(int_sum3, w3_low, a6);
            int_sum3 = my_vdotq_s32(int_sum3, w3_high, a7);

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
            sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(int_sum2), s_2);
            sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(int_sum3), s_3);
        }
        if (params->bias.data_ptr) {
            params->C.data_ptr[j] = params->bias.data_ptr[j] + vaddvq_f32(sumv0) + vaddvq_f32(sumv1) +
                                            vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
        }
        else {
            params->C.data_ptr[j] =
                vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
        }
    }

    return NULL;
}

inline static void* gemm_int8_int4_no_offset_over_column_unroll128(void* args) {
    struct a8w4_thread_args* mat_args = (struct a8w4_thread_args*)args;
    const struct matmul_params* params = mat_args->params;
    int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
    const int num_block = k / block_size;
    int TILE_SIZE = mat_args->tile_size;

    // assert((mat_args->end_i - mat_args->start_i) % TILE_SIZE == 0);
    assert(k % TILE_SIZE == 0);
    assert(n % TILE_SIZE == 0);
    // assert(TILE_SIZE % 4 == 0);

    for (int ti = mat_args->start_i; ti < mat_args->end_i; ti += TILE_SIZE) {
        for (int tj = 0; tj < n; tj += TILE_SIZE) {
            for (int i = ti; i < ti + TILE_SIZE; i++) {
                for (int j = tj; j < tj + TILE_SIZE; j++) {
                    float32x4_t sumv0 = vdupq_n_f32(0.0f);
                    float32x4_t sumv1 = vdupq_n_f32(0.0f);
                    float32x4_t sumv2 = vdupq_n_f32(0.0f);
                    float32x4_t sumv3 = vdupq_n_f32(0.0f);
                    const unsigned char* w_start = &params->B.int4_data_ptr[j * k / 2];
                    const signed char* a_start = &params->A.int8_data_ptr[i * k];
                    float* s_a = &params->A_scales[i * k / 32];
                    float* s_w = &params->scales[j * k / 32];

                    const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
                    const int8x16_t offsets = vdupq_n_s8(8);
                    for (int q = 0; q < num_block; q += 4) {
                        int32x4_t int_sum0 = vdupq_n_s32(0);
                        int32x4_t int_sum1 = vdupq_n_s32(0);
                        int32x4_t int_sum2 = vdupq_n_s32(0);
                        int32x4_t int_sum3 = vdupq_n_s32(0);
                        float s_0 = *s_a++ * *s_w++;
                        float s_1 = *s_a++ * *s_w++;
                        float s_2 = *s_a++ * *s_w++;
                        float s_3 = *s_a++ * *s_w++;

                        const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
                        const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
                        const uint8x16_t w2 = vld1q_u8(w_start + 32);  // 32 4bit weight
                        const uint8x16_t w3 = vld1q_u8(w_start + 48);  // 32 4bit weight
                        w_start += 64;

                        // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
                        // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
                        // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
                        // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
                        // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
                        int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                        int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                        int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
                        int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
                        int8x16_t w2_low = vreinterpretq_s8_u8(vandq_u8(w2, mask_low4bit));
                        int8x16_t w2_high = vreinterpretq_s8_u8(vshrq_n_u8(w2, 4));
                        int8x16_t w3_low = vreinterpretq_s8_u8(vandq_u8(w3, mask_low4bit));
                        int8x16_t w3_high = vreinterpretq_s8_u8(vshrq_n_u8(w3, 4));

                        // apply offset
                        w0_low = vsubq_s8(w0_low, offsets);
                        w0_high = vsubq_s8(w0_high, offsets);
                        w1_low = vsubq_s8(w1_low, offsets);
                        w1_high = vsubq_s8(w1_high, offsets);
                        w2_low = vsubq_s8(w2_low, offsets);
                        w2_high = vsubq_s8(w2_high, offsets);
                        w3_low = vsubq_s8(w3_low, offsets);
                        w3_high = vsubq_s8(w3_high, offsets);

                        // load 64 8-bit activation
                        const int8x16_t a0 = vld1q_s8(a_start);
                        const int8x16_t a1 = vld1q_s8(a_start + 16);
                        const int8x16_t a2 = vld1q_s8(a_start + 32);
                        const int8x16_t a3 = vld1q_s8(a_start + 48);
                        const int8x16_t a4 = vld1q_s8(a_start + 64);
                        const int8x16_t a5 = vld1q_s8(a_start + 80);
                        const int8x16_t a6 = vld1q_s8(a_start + 96);
                        const int8x16_t a7 = vld1q_s8(a_start + 112);
                        a_start += 128;

                        // dot product into int32x4_t
                        int_sum0 = my_vdotq_s32(int_sum0, w0_low, a0);
                        int_sum0 = my_vdotq_s32(int_sum0, w0_high, a1);
                        int_sum1 = my_vdotq_s32(int_sum1, w1_low, a2);
                        int_sum1 = my_vdotq_s32(int_sum1, w1_high, a3);
                        int_sum2 = my_vdotq_s32(int_sum2, w2_low, a4);
                        int_sum2 = my_vdotq_s32(int_sum2, w2_high, a5);
                        int_sum3 = my_vdotq_s32(int_sum3, w3_low, a6);
                        int_sum3 = my_vdotq_s32(int_sum3, w3_high, a7);

                        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
                        sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(int_sum2), s_2);
                        sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(int_sum3), s_3);
                    }
                    if (params->bias.data_ptr) {
                        params->C.data_ptr[i * n + j] = params->bias.data_ptr[j] + vaddvq_f32(sumv0) + vaddvq_f32(sumv1) +
                                                        vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
                    }
                    else {
                        params->C.data_ptr[i * n + j] =
                            vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
                    }


                    //////////////////////////////////////////
                    // float32x4_t sumv0 = vdupq_n_f32(0.0f);
                    // float32x4_t sumv1 = vdupq_n_f32(0.0f);
                    // float32x4_t sumv2 = vdupq_n_f32(0.0f);
                    // float32x4_t sumv3 = vdupq_n_f32(0.0f);
                    // const unsigned char* w_start = &params->B.int4_data_ptr[j * k / 2];
                    // const signed char* a_start = &params->A.int8_data_ptr[i * k];
                    // float* s_a = &params->A_scales[i * k / 32];
                    // float* s_w = &params->scales[j * k / 32];

                    // const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
                    // const int8x16_t offsets = vdupq_n_s8(8);
                    // for (int q = 0; q < num_block; q += 4) {
                    //     int32x4_t int_sum0 = vdupq_n_s32(0);
                    //     int32x4_t int_sum1 = vdupq_n_s32(0);
                    //     int32x4_t int_sum2 = vdupq_n_s32(0);
                    //     int32x4_t int_sum3 = vdupq_n_s32(0);
                    //     float s_0 = *s_a++ * *s_w++;
                    //     float s_1 = *s_a++ * *s_w++;
                    //     float s_2 = *s_a++ * *s_w++;
                    //     float s_3 = *s_a++ * *s_w++;

                    //     const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
                    //     const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
                    //     const uint8x16_t w2 = vld1q_u8(w_start + 32);  // 32 4bit weight
                    //     const uint8x16_t w3 = vld1q_u8(w_start + 48);  // 32 4bit weight
                    //     w_start += 64;

                    //     // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
                    //     // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
                    //     // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
                    //     // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
                    //     // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
                    //     int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                    //     int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                    //     int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
                    //     int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
                    //     int8x16_t w2_low = vreinterpretq_s8_u8(vandq_u8(w2, mask_low4bit));
                    //     int8x16_t w2_high = vreinterpretq_s8_u8(vshrq_n_u8(w2, 4));
                    //     int8x16_t w3_low = vreinterpretq_s8_u8(vandq_u8(w3, mask_low4bit));
                    //     int8x16_t w3_high = vreinterpretq_s8_u8(vshrq_n_u8(w3, 4));

                    //     // apply offset
                    //     w0_low = vsubq_s8(w0_low, offsets);
                    //     w0_high = vsubq_s8(w0_high, offsets);
                    //     w1_low = vsubq_s8(w1_low, offsets);
                    //     w1_high = vsubq_s8(w1_high, offsets);
                    //     w2_low = vsubq_s8(w2_low, offsets);
                    //     w2_high = vsubq_s8(w2_high, offsets);
                    //     w3_low = vsubq_s8(w3_low, offsets);
                    //     w3_high = vsubq_s8(w3_high, offsets);

                    //     // load 64 8-bit activation
                    //     const int8x16_t a0 = vld1q_s8(a_start);
                    //     const int8x16_t a1 = vld1q_s8(a_start + 16);
                    //     const int8x16_t a2 = vld1q_s8(a_start + 32);
                    //     const int8x16_t a3 = vld1q_s8(a_start + 48);
                    //     const int8x16_t a4 = vld1q_s8(a_start + 64);
                    //     const int8x16_t a5 = vld1q_s8(a_start + 80);
                    //     const int8x16_t a6 = vld1q_s8(a_start + 96);
                    //     const int8x16_t a7 = vld1q_s8(a_start + 112);
                    //     a_start += 128;

                    //     // dot product into int32x4_t
                    //     int_sum0 = my_vdotq_s32(int_sum0, w0_low, a0);
                    //     int_sum0 = my_vdotq_s32(int_sum0, w0_high, a1);
                    //     int_sum1 = my_vdotq_s32(int_sum1, w1_low, a2);
                    //     int_sum1 = my_vdotq_s32(int_sum1, w1_high, a3);
                    //     int_sum2 = my_vdotq_s32(int_sum2, w2_low, a4);
                    //     int_sum2 = my_vdotq_s32(int_sum2, w2_high, a5);
                    //     int_sum3 = my_vdotq_s32(int_sum3, w3_low, a6);
                    //     int_sum3 = my_vdotq_s32(int_sum3, w3_high, a7);

                    //     sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                    //     sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
                    //     sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(int_sum2), s_2);
                    //     sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(int_sum3), s_3);
                    // }
                    // if (params->bias.data_ptr) {
                    //     params->C.data_ptr[i * n + j] = params->bias.data_ptr[j] + vaddvq_f32(sumv0) + vaddvq_f32(sumv1) +
                    //                                     vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
                    // }
                    // else {
                    //     params->C.data_ptr[i * n + j] =
                    //         vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
                    // }


                    //////////////////////////////////////////
                    // float32x4_t sumv0 = vdupq_n_f32(0.0f);
                    // float32x4_t sumv1 = vdupq_n_f32(0.0f);
                    // const unsigned char* w_start = &params->B.int4_data_ptr[j * k / 2];
                    // const signed char* a_start = &params->A.int8_data_ptr[i * k];
                    // float* s_a = &params->A_scales[i * k / 32];
                    // float* s_w = &params->scales[j * k / 32];

                    // const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
                    // const int8x16_t offsets = vdupq_n_s8(8);
                    // for (int q = 0; q < num_block; q += 2) {
                    //     int32x4_t int_sum0 = vdupq_n_s32(0);
                    //     int32x4_t int_sum1 = vdupq_n_s32(0);
                    //     float s_0 = *s_a++ * *s_w++;
                    //     float s_1 = *s_a++ * *s_w++;

                    //     const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
                    //     const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
                    //     w_start += 32;

                    //     // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
                    //     // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
                    //     // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
                    //     // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
                    //     // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
                    //     int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                    //     int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                    //     int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
                    //     int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));

                    //     // load 64 8-bit activation
                    //     const int8x16_t a0 = vld1q_s8(a_start);
                    //     const int8x16_t a1 = vld1q_s8(a_start + 16);
                    //     const int8x16_t a2 = vld1q_s8(a_start + 32);
                    //     const int8x16_t a3 = vld1q_s8(a_start + 48);
                    //     a_start += 64;

                    //     // apply offset
                    //     w0_low = vsubq_s8(w0_low, offsets);
                    //     w0_high = vsubq_s8(w0_high, offsets);
                    //     w1_low = vsubq_s8(w1_low, offsets);
                    //     w1_high = vsubq_s8(w1_high, offsets);

                    //     // dot product into int32x4_t
                    //     int_sum0 = my_vdotq_s32(int_sum0, w0_low, a0);
                    //     int_sum1 = my_vdotq_s32(int_sum1, w1_low, a2);
                    //     int_sum0 = my_vdotq_s32(int_sum0, w0_high, a1);
                    //     int_sum1 = my_vdotq_s32(int_sum1, w1_high, a3);

                    //     sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                    //     sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
                    // }
                    // params->C.data_ptr[i * n + j] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
                }
            }
        }
    }

    // Leftover rows w/o tiling
    int left_start_i = mat_args->start_i + ((mat_args->end_i - mat_args->start_i) / TILE_SIZE) * TILE_SIZE;
    for (int i = left_start_i; i < mat_args->end_i; i++) {
        for (int j = 0; j < n; j++) {
            float32x4_t sumv0 = vdupq_n_f32(0.0f);
            float32x4_t sumv1 = vdupq_n_f32(0.0f);
            float32x4_t sumv2 = vdupq_n_f32(0.0f);
            float32x4_t sumv3 = vdupq_n_f32(0.0f);
            const unsigned char* w_start = &params->B.int4_data_ptr[j * k / 2];
            const signed char* a_start = &params->A.int8_data_ptr[i * k];
            float* s_a = &params->A_scales[i * k / 32];
            float* s_w = &params->scales[j * k / 32];

            const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
            const int8x16_t offsets = vdupq_n_s8(8);
            for (int q = 0; q < num_block; q += 4) {
                int32x4_t int_sum0 = vdupq_n_s32(0);
                int32x4_t int_sum1 = vdupq_n_s32(0);
                int32x4_t int_sum2 = vdupq_n_s32(0);
                int32x4_t int_sum3 = vdupq_n_s32(0);
                float s_0 = *s_a++ * *s_w++;
                float s_1 = *s_a++ * *s_w++;
                float s_2 = *s_a++ * *s_w++;
                float s_3 = *s_a++ * *s_w++;

                const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
                const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
                const uint8x16_t w2 = vld1q_u8(w_start + 32);  // 32 4bit weight
                const uint8x16_t w3 = vld1q_u8(w_start + 48);  // 32 4bit weight
                w_start += 64;

                // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
                // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
                // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
                // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
                // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
                int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
                int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
                int8x16_t w2_low = vreinterpretq_s8_u8(vandq_u8(w2, mask_low4bit));
                int8x16_t w2_high = vreinterpretq_s8_u8(vshrq_n_u8(w2, 4));
                int8x16_t w3_low = vreinterpretq_s8_u8(vandq_u8(w3, mask_low4bit));
                int8x16_t w3_high = vreinterpretq_s8_u8(vshrq_n_u8(w3, 4));

                // apply offset
                w0_low = vsubq_s8(w0_low, offsets);
                w0_high = vsubq_s8(w0_high, offsets);
                w1_low = vsubq_s8(w1_low, offsets);
                w1_high = vsubq_s8(w1_high, offsets);
                w2_low = vsubq_s8(w2_low, offsets);
                w2_high = vsubq_s8(w2_high, offsets);
                w3_low = vsubq_s8(w3_low, offsets);
                w3_high = vsubq_s8(w3_high, offsets);

                // load 64 8-bit activation
                const int8x16_t a0 = vld1q_s8(a_start);
                const int8x16_t a1 = vld1q_s8(a_start + 16);
                const int8x16_t a2 = vld1q_s8(a_start + 32);
                const int8x16_t a3 = vld1q_s8(a_start + 48);
                const int8x16_t a4 = vld1q_s8(a_start + 64);
                const int8x16_t a5 = vld1q_s8(a_start + 80);
                const int8x16_t a6 = vld1q_s8(a_start + 96);
                const int8x16_t a7 = vld1q_s8(a_start + 112);
                a_start += 128;

                // dot product into int32x4_t
                int_sum0 = my_vdotq_s32(int_sum0, w0_low, a0);
                int_sum0 = my_vdotq_s32(int_sum0, w0_high, a1);
                int_sum1 = my_vdotq_s32(int_sum1, w1_low, a2);
                int_sum1 = my_vdotq_s32(int_sum1, w1_high, a3);
                int_sum2 = my_vdotq_s32(int_sum2, w2_low, a4);
                int_sum2 = my_vdotq_s32(int_sum2, w2_high, a5);
                int_sum3 = my_vdotq_s32(int_sum3, w3_low, a6);
                int_sum3 = my_vdotq_s32(int_sum3, w3_high, a7);

                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
                sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(int_sum2), s_2);
                sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(int_sum3), s_3);
            }
            if (params->bias.data_ptr) {
                params->C.data_ptr[i * n + j] = params->bias.data_ptr[j] + vaddvq_f32(sumv0) + vaddvq_f32(sumv1) +
                                                vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
            }
            else {
                params->C.data_ptr[i * n + j] =
                    vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
            }
        }
    }

    return NULL;
}

inline static void* gemm_int8_int4_no_offset_over_column_unroll128_v2(void* args) {
    struct a8w4_thread_args* mat_args = (struct a8w4_thread_args*)args;
    const struct matmul_params* params = mat_args->params;
    int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
    const int num_block = k / block_size;
    int TILE_SIZE = mat_args->tile_size;

    // assert((mat_args->end_j - mat_args->start_j) % TILE_SIZE == 0);
    assert(k % TILE_SIZE == 0);
    assert(n % TILE_SIZE == 0);
    // assert(TILE_SIZE % 4 == 0);

    for (int ti = 0; ti < m; ti += TILE_SIZE) {
        for (int tj = mat_args->start_j; tj < mat_args->end_j; tj += TILE_SIZE) {
            for (int i = ti; i < ti + TILE_SIZE; i++) {
                for (int j = tj; j < tj + TILE_SIZE; j++) {
                    float32x4_t sumv0 = vdupq_n_f32(0.0f);
                    float32x4_t sumv1 = vdupq_n_f32(0.0f);
                    float32x4_t sumv2 = vdupq_n_f32(0.0f);
                    float32x4_t sumv3 = vdupq_n_f32(0.0f);
                    const unsigned char* w_start = &params->B.int4_data_ptr[j * k / 2];
                    const signed char* a_start = &params->A.int8_data_ptr[i * k];
                    float* s_a = &params->A_scales[i * k / 32];
                    float* s_w = &params->scales[j * k / 32];

                    const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
                    const int8x16_t offsets = vdupq_n_s8(8);
                    for (int q = 0; q < num_block; q += 4) {
                        int32x4_t int_sum0 = vdupq_n_s32(0);
                        int32x4_t int_sum1 = vdupq_n_s32(0);
                        int32x4_t int_sum2 = vdupq_n_s32(0);
                        int32x4_t int_sum3 = vdupq_n_s32(0);
                        float s_0 = *s_a++ * *s_w++;
                        float s_1 = *s_a++ * *s_w++;
                        float s_2 = *s_a++ * *s_w++;
                        float s_3 = *s_a++ * *s_w++;

                        const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
                        const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
                        const uint8x16_t w2 = vld1q_u8(w_start + 32);  // 32 4bit weight
                        const uint8x16_t w3 = vld1q_u8(w_start + 48);  // 32 4bit weight
                        w_start += 64;

                        // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
                        // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
                        // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
                        // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
                        // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
                        int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                        int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                        int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
                        int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
                        int8x16_t w2_low = vreinterpretq_s8_u8(vandq_u8(w2, mask_low4bit));
                        int8x16_t w2_high = vreinterpretq_s8_u8(vshrq_n_u8(w2, 4));
                        int8x16_t w3_low = vreinterpretq_s8_u8(vandq_u8(w3, mask_low4bit));
                        int8x16_t w3_high = vreinterpretq_s8_u8(vshrq_n_u8(w3, 4));

                        // apply offset
                        w0_low = vsubq_s8(w0_low, offsets);
                        w0_high = vsubq_s8(w0_high, offsets);
                        w1_low = vsubq_s8(w1_low, offsets);
                        w1_high = vsubq_s8(w1_high, offsets);
                        w2_low = vsubq_s8(w2_low, offsets);
                        w2_high = vsubq_s8(w2_high, offsets);
                        w3_low = vsubq_s8(w3_low, offsets);
                        w3_high = vsubq_s8(w3_high, offsets);

                        // load 64 8-bit activation
                        const int8x16_t a0 = vld1q_s8(a_start);
                        const int8x16_t a1 = vld1q_s8(a_start + 16);
                        const int8x16_t a2 = vld1q_s8(a_start + 32);
                        const int8x16_t a3 = vld1q_s8(a_start + 48);
                        const int8x16_t a4 = vld1q_s8(a_start + 64);
                        const int8x16_t a5 = vld1q_s8(a_start + 80);
                        const int8x16_t a6 = vld1q_s8(a_start + 96);
                        const int8x16_t a7 = vld1q_s8(a_start + 112);
                        a_start += 128;

                        // dot product into int32x4_t
                        int_sum0 = my_vdotq_s32(int_sum0, w0_low, a0);
                        int_sum0 = my_vdotq_s32(int_sum0, w0_high, a1);
                        int_sum1 = my_vdotq_s32(int_sum1, w1_low, a2);
                        int_sum1 = my_vdotq_s32(int_sum1, w1_high, a3);
                        int_sum2 = my_vdotq_s32(int_sum2, w2_low, a4);
                        int_sum2 = my_vdotq_s32(int_sum2, w2_high, a5);
                        int_sum3 = my_vdotq_s32(int_sum3, w3_low, a6);
                        int_sum3 = my_vdotq_s32(int_sum3, w3_high, a7);

                        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
                        sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(int_sum2), s_2);
                        sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(int_sum3), s_3);
                    }
                    if (params->bias.data_ptr) {
                        params->C.data_ptr[i * n + j] = params->bias.data_ptr[j] + vaddvq_f32(sumv0) + vaddvq_f32(sumv1) +
                                                        vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
                    }
                    else {
                        params->C.data_ptr[i * n + j] =
                            vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
                    }
                }
            }
        }
    }

    // // Leftover rows w/o tiling
    int left_start_j = mat_args->start_j + ((mat_args->end_j - mat_args->start_j) / TILE_SIZE) * TILE_SIZE;
    // for (int i = 0; i < m; i++) {
    //     for (int j = left_start_j; i < mat_args->end_j; j++) {
    //         float32x4_t sumv0 = vdupq_n_f32(0.0f);
    //         float32x4_t sumv1 = vdupq_n_f32(0.0f);
    //         float32x4_t sumv2 = vdupq_n_f32(0.0f);
    //         float32x4_t sumv3 = vdupq_n_f32(0.0f);
    //         const unsigned char* w_start = &params->B.int4_data_ptr[j * k / 2];
    //         const signed char* a_start = &params->A.int8_data_ptr[i * k];
    //         float* s_a = &params->A_scales[i * k / 32];
    //         float* s_w = &params->scales[j * k / 32];

    //         const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
    //         const int8x16_t offsets = vdupq_n_s8(8);
    //         for (int q = 0; q < num_block; q += 4) {
    //             int32x4_t int_sum0 = vdupq_n_s32(0);
    //             int32x4_t int_sum1 = vdupq_n_s32(0);
    //             int32x4_t int_sum2 = vdupq_n_s32(0);
    //             int32x4_t int_sum3 = vdupq_n_s32(0);
    //             float s_0 = *s_a++ * *s_w++;
    //             float s_1 = *s_a++ * *s_w++;
    //             float s_2 = *s_a++ * *s_w++;
    //             float s_3 = *s_a++ * *s_w++;

    //             const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
    //             const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
    //             const uint8x16_t w2 = vld1q_u8(w_start + 32);  // 32 4bit weight
    //             const uint8x16_t w3 = vld1q_u8(w_start + 48);  // 32 4bit weight
    //             w_start += 64;

    //             // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
    //             // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
    //             // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
    //             // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
    //             // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
    //             int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
    //             int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
    //             int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
    //             int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
    //             int8x16_t w2_low = vreinterpretq_s8_u8(vandq_u8(w2, mask_low4bit));
    //             int8x16_t w2_high = vreinterpretq_s8_u8(vshrq_n_u8(w2, 4));
    //             int8x16_t w3_low = vreinterpretq_s8_u8(vandq_u8(w3, mask_low4bit));
    //             int8x16_t w3_high = vreinterpretq_s8_u8(vshrq_n_u8(w3, 4));

    //             // apply offset
    //             w0_low = vsubq_s8(w0_low, offsets);
    //             w0_high = vsubq_s8(w0_high, offsets);
    //             w1_low = vsubq_s8(w1_low, offsets);
    //             w1_high = vsubq_s8(w1_high, offsets);
    //             w2_low = vsubq_s8(w2_low, offsets);
    //             w2_high = vsubq_s8(w2_high, offsets);
    //             w3_low = vsubq_s8(w3_low, offsets);
    //             w3_high = vsubq_s8(w3_high, offsets);

    //             // load 64 8-bit activation
    //             const int8x16_t a0 = vld1q_s8(a_start);
    //             const int8x16_t a1 = vld1q_s8(a_start + 16);
    //             const int8x16_t a2 = vld1q_s8(a_start + 32);
    //             const int8x16_t a3 = vld1q_s8(a_start + 48);
    //             const int8x16_t a4 = vld1q_s8(a_start + 64);
    //             const int8x16_t a5 = vld1q_s8(a_start + 80);
    //             const int8x16_t a6 = vld1q_s8(a_start + 96);
    //             const int8x16_t a7 = vld1q_s8(a_start + 112);
    //             a_start += 128;

    //             // dot product into int32x4_t
    //             int_sum0 = my_vdotq_s32(int_sum0, w0_low, a0);
    //             int_sum0 = my_vdotq_s32(int_sum0, w0_high, a1);
    //             int_sum1 = my_vdotq_s32(int_sum1, w1_low, a2);
    //             int_sum1 = my_vdotq_s32(int_sum1, w1_high, a3);
    //             int_sum2 = my_vdotq_s32(int_sum2, w2_low, a4);
    //             int_sum2 = my_vdotq_s32(int_sum2, w2_high, a5);
    //             int_sum3 = my_vdotq_s32(int_sum3, w3_low, a6);
    //             int_sum3 = my_vdotq_s32(int_sum3, w3_high, a7);

    //             sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
    //             sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
    //             sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(int_sum2), s_2);
    //             sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(int_sum3), s_3);
    //         }
    //         if (params->bias.data_ptr) {
    //             params->C.data_ptr[i * n + j] = params->bias.data_ptr[j] + vaddvq_f32(sumv0) + vaddvq_f32(sumv1) +
    //                                             vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
    //         }
    //         else {
    //             params->C.data_ptr[i * n + j] =
    //                 vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
    //         }
    //     }
    // }

    return NULL;
}

inline static void* matmul_int8_int4_no_offset_over_column_unroll128(void* args) {
    struct a8w4_thread_args* mat_args = (struct a8w4_thread_args*)args;
    const struct matmul_params* params = mat_args->params;
    int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
    const int num_block = k / block_size;

    for (int i = 0; i < m; i++) {
        for (int j = mat_args->start_j; j < mat_args->end_j; j++) {
            float32x4_t sumv0 = vdupq_n_f32(0.0f);
            float32x4_t sumv1 = vdupq_n_f32(0.0f);
            float32x4_t sumv2 = vdupq_n_f32(0.0f);
            float32x4_t sumv3 = vdupq_n_f32(0.0f);
            const unsigned char* w_start = &params->B.int4_data_ptr[j * k / 2];
            const signed char* a_start = &params->A.int8_data_ptr[i * k];
            float* s_a = &params->A_scales[i * k / 32];
            float* s_w = &params->scales[j * k / 32];
            
            const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
            const int8x16_t offsets = vdupq_n_s8(8);
            for (int q = 0; q < num_block; q += 4) {
                int32x4_t int_sum0 = vdupq_n_s32(0);
                int32x4_t int_sum1 = vdupq_n_s32(0);
                int32x4_t int_sum2 = vdupq_n_s32(0);
                int32x4_t int_sum3 = vdupq_n_s32(0);
                float s_0 = *s_a++ * *s_w++;
                float s_1 = *s_a++ * *s_w++;
                float s_2 = *s_a++ * *s_w++;
                float s_3 = *s_a++ * *s_w++;

                const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
                const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
                const uint8x16_t w2 = vld1q_u8(w_start + 32);  // 32 4bit weight
                const uint8x16_t w3 = vld1q_u8(w_start + 48);  // 32 4bit weight
                w_start += 64;

                // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
                // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
                // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
                // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
                // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
                int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
                int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
                int8x16_t w2_low = vreinterpretq_s8_u8(vandq_u8(w2, mask_low4bit));
                int8x16_t w2_high = vreinterpretq_s8_u8(vshrq_n_u8(w2, 4));
                int8x16_t w3_low = vreinterpretq_s8_u8(vandq_u8(w3, mask_low4bit));
                int8x16_t w3_high = vreinterpretq_s8_u8(vshrq_n_u8(w3, 4));

                // // Sequential data
                // uint8x8_t hi_mask = vdup_n_u8(0xF0);  // 11110000
                // uint8x8_t lo_mask = vdup_n_u8(0x0F);  // 00001111
                // // Split the data into two 64-bit halves
                // uint8x8_t w0_lower_half = vget_low_u8(w0);
                // uint8x8_t w0_upper_half = vget_high_u8(w0);
                // uint8x8_t w0_low_hi_nibbles = vshr_n_u8(vand_u8(w0_lower_half, hi_mask), 4);
                // uint8x8_t w0_low_lo_nibbles = vand_u8(w0_lower_half, lo_mask);
                // int8x16_t w0_low = vreinterpretq_s8_u8(vcombine_u8(w0_low_lo_nibbles, w0_low_hi_nibbles));
                // uint8x8_t w0_high_hi_nibbles = vshr_n_u8(vand_u8(w0_upper_half, hi_mask), 4);
                // uint8x8_t w0_high_lo_nibbles = vand_u8(w0_upper_half, lo_mask);
                // int8x16_t w0_high = vreinterpretq_s8_u8(vcombine_u8(w0_high_lo_nibbles, w0_high_hi_nibbles));

                // uint8x8_t w1_lower_half = vget_low_u8(w1);
                // uint8x8_t w1_upper_half = vget_high_u8(w1);
                // uint8x8_t w1_low_hi_nibbles = vshr_n_u8(vand_u8(w1_lower_half, hi_mask), 4);
                // uint8x8_t w1_low_lo_nibbles = vand_u8(w1_lower_half, lo_mask);
                // int8x16_t w1_low = vreinterpretq_s8_u8(vcombine_u8(w1_low_lo_nibbles, w1_low_hi_nibbles));
                // uint8x8_t w1_high_hi_nibbles = vshr_n_u8(vand_u8(w1_upper_half, hi_mask), 4);
                // uint8x8_t w1_high_lo_nibbles = vand_u8(w1_upper_half, lo_mask);
                // int8x16_t w1_high = vreinterpretq_s8_u8(vcombine_u8(w1_high_lo_nibbles, w1_high_hi_nibbles));

                // uint8x8_t w2_lower_half = vget_low_u8(w2);
                // uint8x8_t w2_upper_half = vget_high_u8(w2);
                // uint8x8_t w2_low_hi_nibbles = vshr_n_u8(vand_u8(w2_lower_half, hi_mask), 4);
                // uint8x8_t w2_low_lo_nibbles = vand_u8(w2_lower_half, lo_mask);
                // int8x16_t w2_low = vreinterpretq_s8_u8(vcombine_u8(w2_low_lo_nibbles, w2_low_hi_nibbles));
                // uint8x8_t w2_high_hi_nibbles = vshr_n_u8(vand_u8(w2_upper_half, hi_mask), 4);
                // uint8x8_t w2_high_lo_nibbles = vand_u8(w2_upper_half, lo_mask);
                // int8x16_t w2_high = vreinterpretq_s8_u8(vcombine_u8(w2_high_lo_nibbles, w2_high_hi_nibbles));

                // uint8x8_t w3_lower_half = vget_low_u8(w3);
                // uint8x8_t w3_upper_half = vget_high_u8(w3);
                // uint8x8_t w3_low_hi_nibbles = vshr_n_u8(vand_u8(w3_lower_half, hi_mask), 4);
                // uint8x8_t w3_low_lo_nibbles = vand_u8(w3_lower_half, lo_mask);
                // int8x16_t w3_low = vreinterpretq_s8_u8(vcombine_u8(w3_low_lo_nibbles, w3_low_hi_nibbles));
                // uint8x8_t w3_high_hi_nibbles = vshr_n_u8(vand_u8(w3_upper_half, hi_mask), 4);
                // uint8x8_t w3_high_lo_nibbles = vand_u8(w3_upper_half, lo_mask);
                // int8x16_t w3_high = vreinterpretq_s8_u8(vcombine_u8(w3_high_lo_nibbles, w3_high_hi_nibbles));

                // apply offset
                w0_low = vsubq_s8(w0_low, offsets);
                w0_high = vsubq_s8(w0_high, offsets);
                w1_low = vsubq_s8(w1_low, offsets);
                w1_high = vsubq_s8(w1_high, offsets);
                w2_low = vsubq_s8(w2_low, offsets);
                w2_high = vsubq_s8(w2_high, offsets);
                w3_low = vsubq_s8(w3_low, offsets);
                w3_high = vsubq_s8(w3_high, offsets);

                // load 64 8-bit activation
                const int8x16_t a0 = vld1q_s8(a_start);
                const int8x16_t a1 = vld1q_s8(a_start + 16);
                const int8x16_t a2 = vld1q_s8(a_start + 32);
                const int8x16_t a3 = vld1q_s8(a_start + 48);
                const int8x16_t a4 = vld1q_s8(a_start + 64);
                const int8x16_t a5 = vld1q_s8(a_start + 80);
                const int8x16_t a6 = vld1q_s8(a_start + 96);
                const int8x16_t a7 = vld1q_s8(a_start + 112);
                a_start += 128;

                // dot product into int32x4_t
                int_sum0 = my_vdotq_s32(int_sum0, w0_low, a0);
                int_sum0 = my_vdotq_s32(int_sum0, w0_high, a1);
                int_sum1 = my_vdotq_s32(int_sum1, w1_low, a2);
                int_sum1 = my_vdotq_s32(int_sum1, w1_high, a3);
                int_sum2 = my_vdotq_s32(int_sum2, w2_low, a4);
                int_sum2 = my_vdotq_s32(int_sum2, w2_high, a5);
                int_sum3 = my_vdotq_s32(int_sum3, w3_low, a6);
                int_sum3 = my_vdotq_s32(int_sum3, w3_high, a7);

                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
                sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(int_sum2), s_2);
                sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(int_sum3), s_3);
            }
            if (params->bias.data_ptr) {
                params->C.data_ptr[i * n + j] = params->bias.data_ptr[j] + vaddvq_f32(sumv0) + vaddvq_f32(sumv1) +
                                                vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
            }
            else {
                params->C.data_ptr[i * n + j] =
                    vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
            }
        }
    }

    return NULL;
}
// inline static void* matmul_int8_int4_no_offset_over_column_unroll128(void* args) {
//     struct a8w4_thread_args* mat_args = (struct a8w4_thread_args*)args;
//     const struct matmul_params* params = mat_args->params;
//     int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
//     const int num_block = k / block_size;

//     for (int i = 0; i < m; i++) {
//         for (int j = mat_args->start_j; j < mat_args->end_j; j++) {
//             float32x4_t sumv0 = vdupq_n_f32(0.0f);
//             float32x4_t sumv1 = vdupq_n_f32(0.0f);
//             float32x4_t sumv2 = vdupq_n_f32(0.0f);
//             float32x4_t sumv3 = vdupq_n_f32(0.0f);
//             float32x4_t sumv4 = vdupq_n_f32(0.0f);
//             float32x4_t sumv5 = vdupq_n_f32(0.0f);
//             float32x4_t sumv6 = vdupq_n_f32(0.0f);
//             float32x4_t sumv7 = vdupq_n_f32(0.0f);
//             const unsigned char* w_start = &params->B.int4_data_ptr[j * k / 2];
//             const signed char* a_start = &params->A.int8_data_ptr[i * k];
//             float* s_a = &params->A_scales[i * k / 32];
//             float* s_w = &params->scales[j * k / 32];

//             const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
//             const int8x16_t offsets = vdupq_n_s8(8);
//             for (int q = 0; q < num_block; q += 8) {
//                 int32x4_t int_sum0 = vdupq_n_s32(0);
//                 int32x4_t int_sum1 = vdupq_n_s32(0);
//                 int32x4_t int_sum2 = vdupq_n_s32(0);
//                 int32x4_t int_sum3 = vdupq_n_s32(0);
//                 int32x4_t int_sum4 = vdupq_n_s32(0);
//                 int32x4_t int_sum5 = vdupq_n_s32(0);
//                 int32x4_t int_sum6 = vdupq_n_s32(0);
//                 int32x4_t int_sum7 = vdupq_n_s32(0);
//                 float s_0 = *s_a++ * *s_w++;
//                 float s_1 = *s_a++ * *s_w++;
//                 float s_2 = *s_a++ * *s_w++;
//                 float s_3 = *s_a++ * *s_w++;


//                 const uint8x16_t w0 = vld1q_u8(w_start);       // 32 4bit weight
//                 const uint8x16_t w1 = vld1q_u8(w_start + 16);  // 32 4bit weight
//                 const uint8x16_t w2 = vld1q_u8(w_start + 32);  // 32 4bit weight
//                 const uint8x16_t w3 = vld1q_u8(w_start + 48);  // 32 4bit weight
//                 w_start += 64;

//                 // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
//                 // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
//                 // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
//                 // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
//                 // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
//                 int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
//                 int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
//                 int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
//                 int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));
//                 int8x16_t w2_low = vreinterpretq_s8_u8(vandq_u8(w2, mask_low4bit));
//                 int8x16_t w2_high = vreinterpretq_s8_u8(vshrq_n_u8(w2, 4));
//                 int8x16_t w3_low = vreinterpretq_s8_u8(vandq_u8(w3, mask_low4bit));
//                 int8x16_t w3_high = vreinterpretq_s8_u8(vshrq_n_u8(w3, 4));

//                 // apply offset
//                 w0_low = vsubq_s8(w0_low, offsets);
//                 w0_high = vsubq_s8(w0_high, offsets);
//                 w1_low = vsubq_s8(w1_low, offsets);
//                 w1_high = vsubq_s8(w1_high, offsets);
//                 w2_low = vsubq_s8(w2_low, offsets);
//                 w2_high = vsubq_s8(w2_high, offsets);
//                 w3_low = vsubq_s8(w3_low, offsets);
//                 w3_high = vsubq_s8(w3_high, offsets);

//                 // load 64 8-bit activation
//                 const int8x16_t a0 = vld1q_s8(a_start);
//                 const int8x16_t a1 = vld1q_s8(a_start + 16);
//                 const int8x16_t a2 = vld1q_s8(a_start + 32);
//                 const int8x16_t a3 = vld1q_s8(a_start + 48);
//                 const int8x16_t a4 = vld1q_s8(a_start + 64);
//                 const int8x16_t a5 = vld1q_s8(a_start + 80);
//                 const int8x16_t a6 = vld1q_s8(a_start + 96);
//                 const int8x16_t a7 = vld1q_s8(a_start + 112);
//                 a_start += 128;

//                 // dot product into int32x4_t
//                 int_sum0 = my_vdotq_s32(int_sum0, w0_low, a0);
//                 int_sum0 = my_vdotq_s32(int_sum0, w0_high, a1);
//                 int_sum1 = my_vdotq_s32(int_sum1, w1_low, a2);
//                 int_sum1 = my_vdotq_s32(int_sum1, w1_high, a3);
//                 int_sum2 = my_vdotq_s32(int_sum2, w2_low, a4);
//                 int_sum2 = my_vdotq_s32(int_sum2, w2_high, a5);
//                 int_sum3 = my_vdotq_s32(int_sum3, w3_low, a6);
//                 int_sum3 = my_vdotq_s32(int_sum3, w3_high, a7);

//                 sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
//                 sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
//                 sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(int_sum2), s_2);
//                 sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(int_sum3), s_3);
//             }
//             if (params->bias.data_ptr)
//                 params->C.data_ptr[i * n + j] = params->bias.data_ptr[j] + vaddvq_f32(sumv0) + vaddvq_f32(sumv1) +
//                                                 vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
//             else
//                 params->C.data_ptr[i * n + j] =
//                     vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
//         }
//     }

//     return NULL;
// }

static void* matmul_int8_int4_no_offset_over_column_packed(void* args) {
    struct a8w4_thread_args* mat_args = (struct a8w4_thread_args*)args;
    const struct matmul_params* params = mat_args->params;
    int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
    const int num_block = k / block_size;

    struct pack_q4_tensor* w_ptr = (struct pack_q4_tensor*)params->B.int4_data_ptr;
    struct pack_q8_tensor* a_ptr = (struct pack_q8_tensor*)params->A.int8_data_ptr;

    for (int i = 0; i < m; i++) {
        for (int j = mat_args->start_j; j < mat_args->end_j; j++) {
            float32x4_t sumv0 = vdupq_n_f32(0.0f);
            float32x4_t sumv1 = vdupq_n_f32(0.0f);

            const uint8x16_t mask_low4bit = vdupq_n_u8(0xf);
            const int8x16_t offsets = vdupq_n_s8(8);

            struct pack_q4_tensor* w_start = w_ptr + j * num_block;
            struct pack_q8_tensor* a_start = a_ptr + i * num_block;
            for (int q = 0; q < num_block; q += 2) {
                int32x4_t int_sum0 = vdupq_n_s32(0);
                int32x4_t int_sum1 = vdupq_n_s32(0);
                float s_0 = w_start[0].scale * a_start[0].scale;
                float s_1 = w_start[1].scale * a_start[1].scale;

                const uint8x16_t w0 = vld1q_u8(w_start[0].qx);  // 32 4bit weight
                const uint8x16_t w1 = vld1q_u8(w_start[1].qx);  // 32 4bit weight

                // Quantization Method QM_ARM, convert 64 4-bit to 64 8-bit
                // sequential: (0, 1), (2, 3), (4, 5), (6, 7)... : 128 bit
                // expected layout of inB: (0, 16), (1, 17), (2, 18), (3, 19)...
                // low; (0, 0), (1, 0), (2, 0), (3, 0) ...
                // high: (16, 0), (17, 0), (18, 0), (19, 0) ...
                int8x16_t w0_low = vreinterpretq_s8_u8(vandq_u8(w0, mask_low4bit));
                int8x16_t w0_high = vreinterpretq_s8_u8(vshrq_n_u8(w0, 4));
                int8x16_t w1_low = vreinterpretq_s8_u8(vandq_u8(w1, mask_low4bit));
                int8x16_t w1_high = vreinterpretq_s8_u8(vshrq_n_u8(w1, 4));

                // load 64 8-bit activation
                const int8x16_t a0 = vld1q_s8(a_start[0].qx);
                const int8x16_t a1 = vld1q_s8(a_start[0].qx + 16);
                const int8x16_t a2 = vld1q_s8(a_start[1].qx);
                const int8x16_t a3 = vld1q_s8(a_start[1].qx + 16);

                a_start += 2;
                w_start += 2;

                // apply offset
                w0_low = vsubq_s8(w0_low, offsets);
                w0_high = vsubq_s8(w0_high, offsets);
                w1_low = vsubq_s8(w1_low, offsets);
                w1_high = vsubq_s8(w1_high, offsets);

                // dot product into int32x4_t
                int_sum0 = my_vdotq_s32(int_sum0, w0_low, a0);
                int_sum1 = my_vdotq_s32(int_sum1, w1_low, a2);
                int_sum0 = my_vdotq_s32(int_sum0, w0_high, a1);
                int_sum1 = my_vdotq_s32(int_sum1, w1_high, a3);

                sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(int_sum0), s_0);
                sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(int_sum1), s_1);
            }
            params->C.data_ptr[i * n + j] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
        }
    }

    return NULL;
}

#ifdef USE_ACCELERATE
inline static void* fp32_matmul_transposed_cblas_gemm(void* args) {
    struct a8w4_thread_args* mat_args = (struct a8w4_thread_args*)args;
    const struct matmul_params* params = mat_args->params;

    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float *data_A = A->data_ptr + mat_args->start_j * A->column;
    float *data_B = B->data_ptr;
    float *data_C = C->data_ptr + mat_args->start_j * C->column;
    float alpha = params->alpha;

    int n = C->column, k = A->column;
    int m = mat_args->end_j - mat_args->start_j;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k,
                alpha, data_A, k,
                       data_B, k,
                0.0f,  data_C, n);
    
    return NULL;
}
#endif

namespace matmul {
void MatmulOperator::mat_mul_accelerator_int8_int4_fast_no_offset(struct matmul_params* params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    assert(params->block_size % 32 == 0);  // support block size to be multiply of 32
    assert(A->row == C->row);              // support block size to be multiply of 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);
    // ref imp.
    // matmul_int8_int4_no_offset(params);

    // const int num_thread = 8;
    const int num_thread = params->opt_params.num_thread;
    // pthread_t thread_pool[num_thread];
    struct a8w4_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now

#ifdef PACK_QK
    // This may lead to performance degradation
    static void *pool = pool_start(matmul_int8_int4_no_offset_over_column_packed, num_thread);
#else
    static void *pool = pool_start(matmul_int8_int4_no_offset_over_column_unroll128, num_thread);
#endif

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_j = j * (params->C.column / num_thread);
        // threads_args[j].end_j = (j + 1) * (params->C.column / num_thread);
        if (j == num_thread - 1) {
            threads_args[j].end_j = params->C.column;
        } else {
            threads_args[j].end_j = (j + 1) * (params->C.column / num_thread);
        }
        threads_args[j].params = params;
// #ifdef PACK_QK
//         // This may lead to performance degradation
//         pthread_create(&thread_pool[j], NULL, matmul_int8_int4_no_offset_over_column_packed, &threads_args[j]);
// #else
//         pthread_create(&thread_pool[j], NULL, matmul_int8_int4_no_offset_over_column_unroll128, &threads_args[j]);
// #endif
        pool_enqueue(pool, &threads_args[j], '\0');
    }
    // Join threads
    // for (j = 0; j < num_thread; j++) pthread_join(thread_pool[j], NULL);
    pool_wait(pool);
};

void MatmulOperator::gemv_accelerator_int8_int4_fast_no_offset(struct matmul_params* params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;
    assert(params->block_size % 32 == 0);  // support block size to be multiply of 32
    assert(A->row == C->row);              // support block size to be multiply of 32
    assert(A->row == 1);

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    const int num_thread = params->opt_params.num_thread;
    struct a8w4_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now

    static void *pool = pool_start(gemv_int8_int4_no_offset_over_column_unroll128, num_thread);

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_j = j * (params->C.column / num_thread);
        if (j == num_thread - 1) {
            threads_args[j].end_j = params->C.column;
        } else {
            threads_args[j].end_j = (j + 1) * (params->C.column / num_thread);
        }
        threads_args[j].params = params;
        pool_enqueue(pool, &threads_args[j], '\0');
    }
    // Join threads
    pool_wait(pool);
};

void MatmulOperator::gemm_accelerator_int8_int4_fast_no_offset(struct matmul_params* params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;
    assert(params->block_size % 32 == 0);  // support block size to be multiply of 32
    assert(A->row == C->row);              // support block size to be multiply of 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    const int num_thread = params->opt_params.num_thread;
    struct a8w4_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now

    static void *pool = pool_start(gemm_int8_int4_no_offset_over_column_unroll128, num_thread);

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_i = j * (params->C.row / num_thread);
        if (j == num_thread - 1) {
            threads_args[j].end_i = params->C.row;
        } else {
            threads_args[j].end_i = (j + 1) * (params->C.row / num_thread);
        }
        threads_args[j].tile_size = 4;
        threads_args[j].params = params;
        pool_enqueue(pool, &threads_args[j], '\0');
    }
    // Join threads
    pool_wait(pool);
};

void MatmulOperator::gemm_accelerator_int8_int4_fast_no_offset_v2(struct matmul_params* params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;
    assert(params->block_size % 32 == 0);  // support block size to be multiply of 32
    assert(A->row == C->row);              // support block size to be multiply of 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    const int num_thread = params->opt_params.num_thread;
    struct a8w4_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now

    static void *pool = pool_start(gemm_int8_int4_no_offset_over_column_unroll128_v2, num_thread);

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_j = j * (params->C.column / num_thread);
        if (j == num_thread - 1) {
            threads_args[j].end_j = params->C.column;
        } else {
            threads_args[j].end_j = (j + 1) * (params->C.column / num_thread);
        }
        threads_args[j].tile_size = 4;
        threads_args[j].params = params;
        pool_enqueue(pool, &threads_args[j], '\0');
    }
    // Join threads
    pool_wait(pool);
};

#ifdef USE_ACCELERATE
void MatmulOperator::cblas_gemm_accelerator_no_offset(struct matmul_params* params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;
    assert(params->block_size % 32 == 0);  // support block size to be multiply of 32
    assert(A->row == C->row);              // support block size to be multiply of 32

    dequantize_int4_to_fp32(B->int4_data_ptr, B->data_ptr, params->scales, A->column * C->column, block_size);

    const int num_thread = params->opt_params.num_thread;
    struct a8w4_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now

    // mat_mul_accelerator_transposed_fastover_column(params);

    static void *pool = pool_start(fp32_matmul_transposed_cblas_gemm, num_thread);

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_j = j * (params->C.row / num_thread);
        if (j == num_thread - 1) {
            threads_args[j].end_j = params->C.row;
        } else {
            threads_args[j].end_j = (j + 1) * (params->C.row / num_thread);
        }
        // &params->A.data_ptr = threads_args[j].start_j * params->A.column;
        // params->A.row = threads_args[j].end_j - threads_args[j].start_j;
        // &params->C.data_ptr = threads_args[j].start_j * params->C.column;
        // params->C.row = threads_args[j].end_j - threads_args[j].start_j;
        // threads_args[j].tile_size = 4;
        threads_args[j].params = params;
        pool_enqueue(pool, &threads_args[j], '\0');
    }
    // Join threads
    pool_wait(pool);
};
#endif

}  // namespace matmul
