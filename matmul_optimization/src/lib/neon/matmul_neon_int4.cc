#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>
#include <arm_neon.h>

#include "../matmul.h"

static void dequantize_block_q4(const uint8_t *int4_w, float *y, float scale, float offset, int block_size) {
    const float32x4_t vd = vdupq_n_f32(scale);
    const float32x4_t vm = vdupq_n_f32(offset);

    const uint8_t *pp = int4_w;

    // for (int qi = 0; qi < block_size/2; qi++) {
    //     uint8_t packed_int4 = int4_w[qi];
    //     float deq_0 = (float)(packed_int4 & 0x0F) * scale + offset;
    //     float deq_1 = (float)(packed_int4 >> 4) * scale + offset;
    //     *y++ = deq_0;
    //     *y++ = deq_1;
    // }

    for (int l = 0; l < block_size; l += 16) {
        // Load 16x4-bit integers into 8x8-bit integers
        const uint8x8_t v8 = vld1_u8(pp + l/2);

        // Expand 4-bit qs to 8-bit bytes
        const uint8x8_t v0 = vand_u8(v8, vdup_n_u8(0x0f));
        const uint8x8_t v1 = vshr_n_u8(v8, 4);

        // Convert to signed 8-bit integers
        const int8x8_t vs_0 = vreinterpret_s8_u8(v0);
        const int8x8_t vs_1 = vreinterpret_s8_u8(v1);

        // Interleave and combine
        const int8x8_t vx_0 = vzip1_s8(vs_0, vs_1);
        const int8x8_t vx_1 = vzip2_s8(vs_0, vs_1);

        const int8x16_t vq = vcombine_s8(vx_0, vx_1);

        // convert to 2x int16x8_t
        const int16x8_t vi_0 = vmovl_s8(vget_low_s8 (vq));
        const int16x8_t vi_1 = vmovl_s8(vget_high_s8(vq));

        // convert to 4x float32x4_t
        const float32x4_t vf_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16 (vi_0)));
        const float32x4_t vf_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vi_0)));
        const float32x4_t vf_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16 (vi_1)));
        const float32x4_t vf_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vi_1)));

        // multiply by d and add m
        const float32x4_t r0 = vmlaq_f32(vm, vf_0, vd);
        const float32x4_t r1 = vmlaq_f32(vm, vf_1, vd);
        const float32x4_t r2 = vmlaq_f32(vm, vf_2, vd);
        const float32x4_t r3 = vmlaq_f32(vm, vf_3, vd);

        // Store
        vst1q_f32(y + l +  0, r0);
        vst1q_f32(y + l +  4, r1);
        vst1q_f32(y + l +  8, r2);
        vst1q_f32(y + l + 12, r3);
    }
}

struct int4_thread_args {
    int start_j, end_j;
    const struct matmul_params *params;
};

static void *fast_over_column_func_v2(void *args) {
    int i, j, k;
    struct int4_thread_args *mat_args = (struct int4_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    float weight_block[4]; // 128 bitwidth

    for (i = 0; i < C->row; i++) {
        for (j = mat_args->start_j; j < mat_args->end_j; j++) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            for (k = 0; k < B->row; k += block_size) {
                float s = scale[j * (B->row / 16) + k / 32];  // /16:B->column is packed 4bits
                float o = offset[j * (B->row / 16) + k / 32];
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->row + k / 2];
                float32x4_t *x_ptr = (float32x4_t *)&A->data_ptr[i * A->column + k];

                float32x4_t weight_4x32, weight_4x32_2;
                float *weight_block = (float *)&weight_4x32, *weight_block2 = (float *)&weight_4x32_2;
                const float32x4_t vs = vdupq_n_f32(s);
                const float32x4_t vo = vdupq_n_f32(o);
                int qi = 0;
                for (int l = 0; l < block_size/8; l++){
                    uint8_t packed_int4 = weight_32_int4[qi++];
                    weight_block[0] = (float)(packed_int4 & 0x0F);
                    weight_block[1] = (float)(packed_int4 >> 4);
                    packed_int4 = weight_32_int4[qi++];
                    weight_block[2] = (float)(packed_int4 & 0x0F);
                    weight_block[3] = (float)(packed_int4 >> 4);
                    weight_4x32 = vmlaq_f32(vo, weight_4x32, vs);
                    packed_int4 = weight_32_int4[qi++];
                    weight_block2[0] = (float)(packed_int4 & 0x0F);
                    weight_block2[1] = (float)(packed_int4 >> 4);
                    packed_int4 = weight_32_int4[qi++];
                    weight_block2[2] = (float)(packed_int4 & 0x0F);
                    weight_block2[3] = (float)(packed_int4 >> 4);
                    weight_4x32_2 = vmlaq_f32(vo, weight_4x32_2, vs);

                    acc0 = vaddq_f32(vmulq_f32(*x_ptr++, weight_4x32), acc0);
                    acc0 = vaddq_f32(vmulq_f32(*x_ptr++, weight_4x32_2), acc0);
                }   
            }
            float *ptr = (float *)&acc0;
            C->data_ptr[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3];
        }
    }

    return NULL;
}

static void *fast_over_column_func_v1(void *args) {
    int i, j, k;
    struct int4_thread_args *mat_args = (struct int4_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    float weight_block[block_size];

    for (i = 0; i < C->row; i++) {
        for (j = mat_args->start_j; j < mat_args->end_j; j++) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            for (k = 0; k < B->row; k += block_size) {
                float s = scale[j * (B->row / 16) + k / 32];  // /16:B->column is packed 4bits
                float o = offset[j * (B->row / 16) + k / 32];
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->row + k / 2];
                dequantize_block_q4(weight_32_int4, weight_block, s, o, block_size);
                float32x4_t *x_ptr = (float32x4_t *)&A->data_ptr[i * A->column + k];
                float32x4_t *w_ptr = (float32x4_t *)&weight_block;

                // assume block_size == 32 (8 x 4 float)
                acc0 = vaddq_f32(vmulq_f32(*x_ptr++, *w_ptr++), acc0);
                acc0 = vaddq_f32(vmulq_f32(*x_ptr++, *w_ptr++), acc0);
                acc0 = vaddq_f32(vmulq_f32(*x_ptr++, *w_ptr++), acc0);
                acc0 = vaddq_f32(vmulq_f32(*x_ptr++, *w_ptr++), acc0);
                acc0 = vaddq_f32(vmulq_f32(*x_ptr++, *w_ptr++), acc0);
                acc0 = vaddq_f32(vmulq_f32(*x_ptr++, *w_ptr++), acc0);
                acc0 = vaddq_f32(vmulq_f32(*x_ptr++, *w_ptr++), acc0);
                acc0 = vaddq_f32(vmulq_f32(*x_ptr++, *w_ptr++), acc0);
            }
            float *ptr = (float *)&acc0;
            C->data_ptr[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3];
        }
    }

    return NULL;
}

namespace matmul {

void MatmulOperator::mat_mul_accelerator_int4_fast(const struct matmul_params *params) {
    const int num_thread = 16;
    int i, j, k;
    pthread_t thread_pool[num_thread];
    struct int4_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_j = j * (params->C.column / num_thread);
        threads_args[j].end_j = (j + 1) * (params->C.column / num_thread);
        threads_args[j].params = params;
        pthread_create(&thread_pool[j], NULL, fast_over_column_func_v1, &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) pthread_join(thread_pool[j], NULL);

};

}  // namespace matmul
