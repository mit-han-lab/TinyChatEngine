#include <immintrin.h>
#include <pthread.h>

#include <cassert>
#include <cmath>

#include "matmul_avx.h"

#include "pthread_pool.h"

struct int4_thread_args {
    int start_j, end_j;
    const struct matmul_params *params;
};

#define FP16_TO_FP32(x) ((float) (x))
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
    // Get absolute values of x vectors
    const __m128i ax = _mm_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m128i sy = _mm_sign_epi8(y, x);
    // Perform multiplication and create 16-bit values
    const __m128i dot = _mm_maddubs_epi16(ax, sy);
    const __m128i ones = _mm_set1_epi16(1);
    return _mm_madd_epi16(ones, dot);
}

// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// horizontally add 4 int32_t
static inline int hsum_i32_4(const __m128i a) {
    const __m128i hi64 = _mm_unpackhi_epi64(a, a);
    const __m128i sum64 = _mm_add_epi32(hi64, a);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(
            0x0303030303030303, 0x0202020202020202,
            0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    return _mm256_and_si256(lowMask, bytes);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if __AVXVNNI__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float(ax, sy);
#endif
}

static inline __m128i packNibbles( __m256i bytes )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
#if __AVX512F__
    const __m256i bytes_srli_4 = _mm256_srli_epi16(bytes, 4);   // 0000_0000_abcd_0000
    bytes = _mm256_or_si256(bytes, bytes_srli_4);               // 0000_abcd_abcd_efgh
    return _mm256_cvtepi16_epi8(bytes);                         // abcd_efgh
#else
    const __m256i lowByte = _mm256_set1_epi16( 0xFF );
    __m256i high = _mm256_andnot_si256( lowByte, bytes );
    __m256i low = _mm256_and_si256( lowByte, bytes );
    high = _mm256_srli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );

    // Compress uint16_t lanes into bytes
    __m128i r0 = _mm256_castsi256_si128( bytes );
    __m128i r1 = _mm256_extracti128_si256( bytes, 1 );
    return _mm_packus_epi16( r0, r1 );
#endif
}

// static inline void merge_int4_int8_dot_product_unroll2block(float *s, float *s_a, uint8_t *w_ptr, __m256i *x_ptr,
//                                                             __m256 &acc0) {
//     // load 0 - 127 bit and 128 - 255
//     __m128i raw_w_0 = _mm_loadu_si128((const __m128i *)w_ptr);
//     __m128i raw_w_128 = _mm_loadu_si128((const __m128i *)(w_ptr + 16));

//     __m256 v_s = _mm256_set1_ps(s[0] * s_a[0]);
//     __m256 v_s2 = _mm256_set1_ps(s[1] * s_a[1]);

//     __m256i activation = x_ptr[0];
//     __m256i activation2 = x_ptr[1];

//     // Expand bytes into uint16_t values
//     __m256i w_8_16exp = _mm256_cvtepu8_epi16(raw_w_0);
//     __m256i w2_8_16exp = _mm256_cvtepu8_epi16(raw_w_128);

//     // Unpack values into individual bytes
//     __m256i raw_w = _mm256_loadu_si256((const __m256i *)w_ptr);
//     const __m256i lowMask = _mm256_set1_epi8(0xF);
//     __m256i w_0 = _mm256_and_si256(lowMask, raw_w);
//     __m256i high = _mm256_andnot_si256(lowMask, raw_w);
//     __m256i w_128 = _mm256_srli_epi16(high, 4);
//     const __m256i zero_point = _mm256_set1_epi8(8);
//     w_0 = _mm256_sub_epi8(w_0, zero_point);
//     w_128 = _mm256_sub_epi8(w_128, zero_point);

//     // Get absolute values of x vectors
//     const __m256i ax = _mm256_sign_epi8(w_0, w_0);
//     const __m256i ax2 = _mm256_sign_epi8(w_128, w_128);
//     // Sign the values of the y vectors
//     const __m256i sy = _mm256_sign_epi8(activation, w_0);
//     const __m256i sy2 = _mm256_sign_epi8(activation2, w_128);
//     // Perform multiplication and create 16-bit values
//     const __m256i dot = _mm256_maddubs_epi16(ax, sy);
//     const __m256i dot2 = _mm256_maddubs_epi16(ax2, sy2);

//     const __m256i ones = _mm256_set1_epi16(1);
//     const __m256i summed_pairs = _mm256_madd_epi16(ones, dot);
//     const __m256i summed_pairs2 = _mm256_madd_epi16(ones, dot2);
//     __m256 intermediate = _mm256_cvtepi32_ps(summed_pairs);
//     __m256 intermediate2 = _mm256_cvtepi32_ps(summed_pairs2);

//     acc0 = _mm256_fmadd_ps(intermediate, v_s, acc0);
//     acc0 = _mm256_fmadd_ps(intermediate2, v_s2, acc0);
// }
inline static void merge_int4_int8_dot_product_unroll2block(float *s, float *s_a, uint8_t *w_ptr, __m256i *x_ptr,
                                                            __m256 &acc0) {
    __m256 v_s = _mm256_set1_ps(s[0] * s_a[0]);
    __m256 v_s2 = _mm256_set1_ps(s[1] * s_a[1]);

    __m256i activation = x_ptr[0];
    __m256i activation2 = x_ptr[1];

    // __m256i w_0 = bytes_from_nibbles_32(w_ptr);
    // Unpack values into individual bytes
    __m256i raw_w = _mm256_loadu_si256((const __m256i *)w_ptr);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    __m256i w_0 = _mm256_and_si256(lowMask, raw_w);

    __m256i high = _mm256_andnot_si256(lowMask, raw_w);
    __m256i w_128 = _mm256_srli_epi16(high, 4);

    const __m256i zero_point = _mm256_set1_epi8(8);
    w_0 = _mm256_sub_epi8(w_0, zero_point);
    w_128 = _mm256_sub_epi8(w_128, zero_point);

    const __m256 intermediate = mul_sum_i8_pairs_float(w_0, activation);
    const __m256 intermediate2 = mul_sum_i8_pairs_float(w_128, activation2);
    // // Get absolute values of x vectors
    // const __m256i ax = _mm256_sign_epi8(w_0, w_0);
    // const __m256i ax2 = _mm256_sign_epi8(w_128, w_128);
    // // Sign the values of the y vectors
    // const __m256i sy = _mm256_sign_epi8(activation, w_0);
    // const __m256i sy2 = _mm256_sign_epi8(activation2, w_128);
    // // Perform multiplication and create 16-bit values
    // const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    // const __m256i dot2 = _mm256_maddubs_epi16(ax2, sy2);

    // const __m256i ones = _mm256_set1_epi16(1);
    // const __m256i summed_pairs = _mm256_madd_epi16(ones, dot);
    // const __m256i summed_pairs2 = _mm256_madd_epi16(ones, dot2);
    // __m256 intermediate = _mm256_cvtepi32_ps(summed_pairs);
    // __m256 intermediate2 = _mm256_cvtepi32_ps(summed_pairs2);

    acc0 = _mm256_fmadd_ps(intermediate, v_s, acc0);
    acc0 = _mm256_fmadd_ps(intermediate2, v_s2, acc0);
}

inline static void *fast_int8_int4_zp_no_offset_over_column_unroll2block(void *args) {
    int i, j, k;
    struct int4_thread_args *mat_args = (struct int4_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    // assume using 8 for now
    const __m256i zero_point = _mm256_set1_epi8(8);
    for (i = 0; i < C->row; i++) {
        for (j = mat_args->start_j; j < mat_args->end_j; j++) {
            __m256 acc0 = _mm256_setzero_ps();
            float *s_ptr = &scale[j * (A->column / block_size)];
            float *sa_ptr = &params->A_scales[(i * A->column) / block_size];
            uint8_t *w_ptr = &B->int4_data_ptr[j * B->row];
            int8_t *x_ptr = &A->int8_data_ptr[i * A->column];
            for (k = 0; k < B->row / block_size; k++) {
                merge_int4_int8_dot_product_unroll2block(s_ptr, sa_ptr, w_ptr, (__m256i *)x_ptr, acc0);
                s_ptr += 2;
                sa_ptr += 2;
                w_ptr += 32;
                x_ptr += 32 * 2;
            }
            float *ptr = (float *)&acc0;
            if (params->bias.data_ptr)
                C->data_ptr[i * C->column + j] =
                    ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7] + params->bias.data_ptr[j];
            else
                // C->data_ptr[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
                C->data_ptr[i * C->column + j] = hsum_float_8(acc0);
        }
    }
    return NULL;
}

static void quantize_fp_to_int8_block_size32(float *x, int size, int8_t *qx, float *qs) {
    int nb = size / 32;
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float maxScalar = _mm_cvtss_f32(max4);

        // Quantize these floats
        const float d = maxScalar / 127.f;
        *qs++ = d;
        const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);

        // Apply the multiplier
        v0 = _mm256_mul_ps(v0, mul);
        v1 = _mm256_mul_ps(v1, mul);
        v2 = _mm256_mul_ps(v2, mul);
        v3 = _mm256_mul_ps(v3, mul);

        // Round to nearest integer
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);

        // Convert int32 to int16
        i0 = _mm256_packs_epi32(i0, i1);  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32(i2, i3);  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                          // Convert int16 to int8
        i0 = _mm256_packs_epi16(i0, i2);  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7,
                                          // 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);

        _mm256_storeu_si256((__m256i *)qx, i0);
        qx += 32;
    }
}

namespace matmul {

void MatmulOperatorAVX::mat_mul_accelerator_int8_int4_fast_no_offset(struct matmul_params *params) {
    // const int num_thread = 4;
    const int num_thread = params->opt_params.num_thread;
    int i, j, k;
    // pthread_t thread_pool[num_thread];
    struct int4_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now
    assert(params->A.column % (params->block_size * 2) == 0);
    // assert((params->C.column % (num_thread * 2)) == 0);  // support block size 32 for now

    // quantize A
    assert((params->A.column * params->A.row) % params->block_size == 0);
    quantize_fp_to_int8_block_size32(params->A.data_ptr, params->A.column * params->A.row, params->A.int8_data_ptr,
                                     params->A_scales);

    static void *pool = pool_start(fast_int8_int4_zp_no_offset_over_column_unroll2block, num_thread);

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_j = j * (params->C.column / num_thread);
        if (j == num_thread - 1) {
            threads_args[j].end_j = params->C.column;
        } else {
            threads_args[j].end_j = (j + 1) * (params->C.column / num_thread);
        }
        threads_args[j].params = params;
        // pthread_create(&thread_pool[j], NULL, fast_int8_int4_zp_no_offset_over_column_unroll2block, &threads_args[j]);
        pool_enqueue(pool, &threads_args[j], '\0');
    }
    // // Join threads
    // for (j = 0; j < num_thread; j++) pthread_join(thread_pool[j], NULL);
    pool_wait(pool);
};
}  // namespace matmul
