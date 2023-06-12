#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>
#include <arm_neon.h>

#include "../matmul.h"


static inline void dequantize_block_q4_unroll2_no_offset(const uint8_t *int4_w, float *y, float scale, const uint8_t *int4_w_2, float *y_2, float scale_2, int block_size) {
    const float32x4_t vd = vdupq_n_f32(scale);
    const float32x4_t vd_2 = vdupq_n_f32(scale_2);
    const float32x4_t zp = vdupq_n_f32(-8.0f);

    for (int l = 0; l < block_size; l += 16) {
        // Load 16x4-bit integers into 8x8-bit integers
        const uint8x8_t v8 = vld1_u8(int4_w + l/2);
        const uint8x8_t v8_2 = vld1_u8(int4_w_2 + l/2);

        // Expand 4-bit qs to 8-bit bytes
        const uint8x8_t v0 = vand_u8(v8, vdup_n_u8(0x0f));
        const uint8x8_t v0_2 = vand_u8(v8_2, vdup_n_u8(0x0f));
        const uint8x8_t v1 = vshr_n_u8(v8, 4);
        const uint8x8_t v1_2 = vshr_n_u8(v8_2, 4);

        // Convert to signed 8-bit integers
        const int8x8_t vs_0 = vreinterpret_s8_u8(v0);
        const int8x8_t vs_0_2 = vreinterpret_s8_u8(v0_2);
        const int8x8_t vs_1 = vreinterpret_s8_u8(v1);
        const int8x8_t vs_1_2 = vreinterpret_s8_u8(v1_2);

        // Interleave and combine
        const int8x8_t vx_0 = vzip1_s8(vs_0, vs_1);
        const int8x8_t vx_0_2 = vzip1_s8(vs_0_2, vs_1_2);
        const int8x8_t vx_1 = vzip2_s8(vs_0, vs_1);
        const int8x8_t vx_1_2 = vzip2_s8(vs_0_2, vs_1_2);

        const int8x16_t vq = vcombine_s8(vx_0, vx_1);
        const int8x16_t vq_2 = vcombine_s8(vx_0_2, vx_1_2);

        // convert to 2x int16x8_t
        auto vq_low = vget_low_s8 (vq);
        auto vq_high = vget_high_s8(vq);
        auto vq_2_low = vget_low_s8 (vq_2);
        auto vq_2_high = vget_high_s8(vq_2);
        const int16x8_t vi_0 = vmovl_s8(vq_low);
        const int16x8_t vi_1 = vmovl_s8(vq_high);
        const int16x8_t vi_2_0 = vmovl_s8(vq_2_low);
        const int16x8_t vi_2_1 = vmovl_s8(vq_2_high);

        // convert to 4x float32x4_t
        auto vi_0_low = vget_low_s16(vi_0);
        auto vi_0_high = vget_high_s16(vi_0);
        auto vi_1_low = vget_low_s16(vi_1);
        auto vi_1_high = vget_high_s16(vi_1);

        auto vi_2_0_low = vget_low_s16(vi_2_0);
        auto vi_2_0_high = vget_high_s16(vi_2_0);
        auto vi_2_1_low = vget_low_s16(vi_2_1);
        auto vi_2_1_high = vget_high_s16(vi_2_1);

        auto vi_0_low_s32 = vmovl_s16(vi_0_low);
        auto vi_0_high_s32 = vmovl_s16(vi_0_high);
        auto vi_1_low_s32 = vmovl_s16(vi_1_low);
        auto vi_1_high_s32 = vmovl_s16(vi_1_high);

        auto vi_2_0_low_s32 = vmovl_s16(vi_2_0_low);
        auto vi_2_0_high_s32 = vmovl_s16(vi_2_0_high);
        auto vi_2_1_low_s32 = vmovl_s16(vi_2_1_low);
        auto vi_2_1_high_s32 = vmovl_s16(vi_2_1_high);

        const float32x4_t vf_0 = vcvtq_f32_s32(vi_0_low_s32);
        const float32x4_t vf_1 = vcvtq_f32_s32(vi_0_high_s32);
        const float32x4_t vf_2 = vcvtq_f32_s32(vi_1_low_s32);
        const float32x4_t vf_3 = vcvtq_f32_s32(vi_1_high_s32);

        const float32x4_t vf_0_2 = vcvtq_f32_s32(vi_2_0_low_s32);
        const float32x4_t vf_1_2 = vcvtq_f32_s32(vi_2_0_high_s32);
        const float32x4_t vf_2_2 = vcvtq_f32_s32(vi_2_1_low_s32);
        const float32x4_t vf_3_2 = vcvtq_f32_s32(vi_2_1_high_s32);

        // (v + (-zp)) * scale
        const float32x4_t r0_zp = vaddq_f32(vf_0, zp);
        const float32x4_t r0_zp2 = vaddq_f32(vf_0_2, zp);
        const float32x4_t r0 = vmulq_f32(r0_zp, vd);
        const float32x4_t r0_2 = vmulq_f32(r0_zp2, vd_2);
        vst1q_f32(y + l + 0, r0);
        vst1q_f32(y_2 + l + 0, r0_2);

        const float32x4_t r1_zp = vaddq_f32(vf_1, zp);
        const float32x4_t r1_zp2 = vaddq_f32(vf_1_2, zp);
        const float32x4_t r1 = vmulq_f32(r1_zp, vd);
        const float32x4_t r1_2 = vmulq_f32(r1_zp2, vd_2);
        vst1q_f32(y + l + 4, r1);
        vst1q_f32(y_2 + l + 4, r1_2);
        
        const float32x4_t r2_zp = vaddq_f32(vf_2, zp);
        const float32x4_t r2_zp2 = vaddq_f32(vf_2_2, zp);
        const float32x4_t r2 = vmulq_f32(r2_zp, vd);
        const float32x4_t r2_2 = vmulq_f32(r2_zp2, vd_2);
        vst1q_f32(y + l + 8, r2);
        vst1q_f32(y_2 + l + 8, r2_2);
        
        const float32x4_t r3_zp = vaddq_f32(vf_3, zp);
        const float32x4_t r3_zp2 = vaddq_f32(vf_3_2, zp);
        const float32x4_t r3 = vmulq_f32(r3_zp, vd);
        const float32x4_t r3_2 = vmulq_f32(r3_zp2, vd_2);
        vst1q_f32(y + l + 12, r3);
        vst1q_f32(y_2 + l + 12, r3_2);
    }
}

static void dequantize_block_q4_no_offset_v1(const uint8_t *int4_w, float *y, float scale, int block_size) {
    const float32x4_t vd = vdupq_n_f32(scale);
    const float32x4_t zp = vdupq_n_f32(-8.0f);
    const auto tmp = vdup_n_u8(0x0f);

    for (int l = 0; l < block_size; l += 16) {
        // Load 16x4-bit integers into 8x8-bit integers
        const uint8x8_t v8 = vld1_u8(int4_w + l/2);

        // Expand 4-bit qs to 8-bit bytes
        const uint8x8_t v0 = vand_u8(v8, tmp);
        const uint8x8_t v1 = vshr_n_u8(v8, 4);

        // Convert to signed 8-bit integers
        const int8x8_t vs_0 = vreinterpret_s8_u8(v0);
        const int8x8_t vs_1 = vreinterpret_s8_u8(v1);

        // Interleave and combine
        const int8x8_t vx_0 = vzip1_s8(vs_0, vs_1);
        const int8x8_t vx_1 = vzip2_s8(vs_0, vs_1);

        const int8x16_t vq = vcombine_s8(vx_0, vx_1);

        // convert to 2x int16x8_t
        auto vq_low = vget_low_s8 (vq);
        auto vq_high = vget_high_s8(vq);
        const int16x8_t vi_0 = vmovl_s8(vq_low);
        const int16x8_t vi_1 = vmovl_s8(vq_high);

        // convert to 4x float32x4_t
        auto vi_0_low = vget_low_s16(vi_0);
        auto vi_0_high = vget_high_s16(vi_0);
        auto vi_1_low = vget_low_s16(vi_1);
        auto vi_1_high = vget_high_s16(vi_1);

        auto vi_0_low_s32 = vmovl_s16(vi_0_low);
        auto vi_0_high_s32 = vmovl_s16(vi_0_high);
        auto vi_1_low_s32 = vmovl_s16(vi_1_low);
        auto vi_1_high_s32 = vmovl_s16(vi_1_high);

        const float32x4_t vf_0 = vcvtq_f32_s32(vi_0_low_s32);
        const float32x4_t vf_1 = vcvtq_f32_s32(vi_0_high_s32);
        const float32x4_t vf_2 = vcvtq_f32_s32(vi_1_low_s32);
        const float32x4_t vf_3 = vcvtq_f32_s32(vi_1_high_s32);

        // (v + (-zp)) * scale
        const float32x4_t r0_zp = vaddq_f32(vf_0, zp);
        const float32x4_t r1_zp = vaddq_f32(vf_1, zp);
        const float32x4_t r2_zp = vaddq_f32(vf_2, zp);
        const float32x4_t r3_zp = vaddq_f32(vf_3, zp);
        const float32x4_t r0 = vmulq_f32(r0_zp, vd);
        const float32x4_t r1 = vmulq_f32(r1_zp, vd);
        const float32x4_t r2 = vmulq_f32(r2_zp, vd);
        const float32x4_t r3 = vmulq_f32(r3_zp, vd);
        vst1q_f32(y + l + 0, r0);
        vst1q_f32(y + l + 4, r1);
        vst1q_f32(y + l + 8, r2);
        vst1q_f32(y + l + 12, r3);
    }
}

struct int4_thread_args {
    int start_j, end_j;
    const struct matmul_params *params;
};

static void *fast_zp_no_offset_over_column_func_v1(void *args) {
    int i, j, k;
    struct int4_thread_args *mat_args = (struct int4_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    float weight_block[block_size];
    float weight_block2[block_size];

    for (i = 0; i < C->row; i++) {
        for (j = mat_args->start_j; j < mat_args->end_j; j+=2) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            for (k = 0; k < B->row * 2; k += block_size) {
                float s = scale[j * (B->row / 16) + k / 32], s1 = scale[(j + 1) * (B->row / 16) + k / 32];
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->row + k / 2], *weight_32_int4_2 = &B->int4_data_ptr[(j+1) * B->row + k / 2];
                dequantize_block_q4_unroll2_no_offset(weight_32_int4, weight_block, s, weight_32_int4_2, weight_block2, s1, block_size);
                float32x4_t *x_ptr = (float32x4_t *)&A->data_ptr[i * A->column + k];
                float32x4_t *w_ptr = (float32x4_t *)&weight_block;
                float32x4_t *w2_ptr = (float32x4_t *)&weight_block2;

                // assume block_size == 32 (8 x 4 float)
                acc0 = vaddq_f32(vmulq_f32(x_ptr[0], w_ptr[0]), acc0);
                acc1 = vaddq_f32(vmulq_f32(x_ptr[0], w2_ptr[0]), acc1);
                acc0 = vaddq_f32(vmulq_f32(x_ptr[1], w_ptr[1]), acc0);
                acc1 = vaddq_f32(vmulq_f32(x_ptr[1], w2_ptr[1]), acc1);
                acc0 = vaddq_f32(vmulq_f32(x_ptr[2], w_ptr[2]), acc0);
                acc1 = vaddq_f32(vmulq_f32(x_ptr[2], w2_ptr[2]), acc1);
                acc0 = vaddq_f32(vmulq_f32(x_ptr[3], w_ptr[3]), acc0);
                acc1 = vaddq_f32(vmulq_f32(x_ptr[3], w2_ptr[3]), acc1);

                acc0 = vaddq_f32(vmulq_f32(x_ptr[4], w_ptr[4]), acc0);
                acc1 = vaddq_f32(vmulq_f32(x_ptr[4], w2_ptr[4]), acc1);
                acc0 = vaddq_f32(vmulq_f32(x_ptr[5], w_ptr[5]), acc0);
                acc1 = vaddq_f32(vmulq_f32(x_ptr[5], w2_ptr[5]), acc1);
                acc0 = vaddq_f32(vmulq_f32(x_ptr[6], w_ptr[6]), acc0);
                acc1 = vaddq_f32(vmulq_f32(x_ptr[6], w2_ptr[6]), acc1);
                acc0 = vaddq_f32(vmulq_f32(x_ptr[7], w_ptr[7]), acc0);
                acc1 = vaddq_f32(vmulq_f32(x_ptr[7], w2_ptr[7]), acc1);
            }
            float *ptr = (float *)&acc0;
            C->data_ptr[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3];
            ptr = (float *)&acc1;
            C->data_ptr[i * C->column + j+1] = ptr[0] + ptr[1] + ptr[2] + ptr[3];
        }
    }

    return NULL;
}

namespace matmul {

void MatmulOperator::mat_mul_accelerator_int4_fast_no_offset(const struct matmul_params *params) {
    const int num_thread = 32;
    int i, j, k;
    pthread_t thread_pool[num_thread];
    struct int4_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_j = j * (params->C.column / num_thread);
        threads_args[j].end_j = (j + 1) * (params->C.column / num_thread);
        threads_args[j].params = params;
        pthread_create(&thread_pool[j], NULL, fast_zp_no_offset_over_column_func_v1, &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) pthread_join(thread_pool[j], NULL);

};

}  // namespace matmul
