#include <immintrin.h>
#include <pthread.h>

#include <cassert>

#include "../matmul.h"

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
static void dequantize_block_q4(const uint8_t *int4_w, float *y, float scale, float offset, int block_size) {
    const __m256 d_v = _mm256_broadcast_ss(&scale);
    const __m256 d_m = _mm256_broadcast_ss(&offset);

    const uint8_t *pp = int4_w;

    for (int l = 0; l < block_size; l += 32) {
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

    float weight_block[block_size];
    float weight_block2[block_size];

    for (i = 0; i < C->row; i++) {
        for (j = mat_args->start_j; j < mat_args->end_j; j += 2) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (k = 0; k < B->row; k += block_size) {
                float s = scale[j * (B->row / 16) + k / 32], s1 = scale[(j + 1) * (B->row / 16) + k / 32];
                float o = offset[j * (B->row / 16) + k / 32], o1 = offset[(j + 1) * (B->row / 16) + k / 32];
                // float zp = zero_point(0, j, k/32);
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->row + k / 2];
                uint8_t *weight_32_int4_2 = &B->int4_data_ptr[(j + 1) * B->row + k / 2];
                __m256 *x_ptr = (__m256 *)&A->data_ptr[i * A->column + k];
                __m256 *w_ptr = (__m256 *)&weight_block;
                __m256 *w2_ptr = (__m256 *)&weight_block2;
                dequantize_block_q4(weight_32_int4, weight_block, s, o, block_size);
                dequantize_block_q4(weight_32_int4_2, weight_block2, s1, o1, block_size);

                // assume block_size == 32 (8 x 32 float)
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
            C->data_ptr[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
            ptr = (float *)&acc1;
            C->data_ptr[i * C->column + j + 1] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
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
            __m256 acc0 = _mm256_setzero_ps();
            for (k = 0; k < B->row; k += block_size) {
                float s = scale[j * (B->row / 16) + k / 32];  // /16:B->row is packed 4bits
                float o = offset[j * (B->row / 16) + k / 32];
                // float zp = zero_point(0, j, k/32);
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->row + k / 2];
                __m256 *x_ptr = (__m256 *)&A->data_ptr[i * A->column + k];
                __m256 *w_ptr = (__m256 *)&weight_block;
                dequantize_block_q4(weight_32_int4, weight_block, s, o, block_size);

                // assume block_size == 32 (8 x 4 float)
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(*x_ptr++, *w_ptr++));
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(*x_ptr++, *w_ptr++));
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(*x_ptr++, *w_ptr++));
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(*x_ptr++, *w_ptr++));
            }
            float *ptr = (float *)&acc0;
            C->data_ptr[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
        }
    }
    return NULL;
}

namespace matmul {
void MatmulOperator::mat_mul_accelerator_int4_fast(const struct matmul_params *params) {
    const int num_thread = params->opt_params.num_thread;
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
