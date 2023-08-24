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

// Dequantize a block of weight
static void dequantize_block_q4_zp_no_offset(const uint8_t *int4_w, float *y, float scale, int block_size) {
    float zp = 8.0f;

    const __m256 d_v = _mm256_broadcast_ss(&scale);
    const __m256 d_zp = _mm256_broadcast_ss(&zp);

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
            const __m256 result = _mm256_mul_ps(_mm256_sub_ps(vf[j], d_zp), d_v);
            _mm256_storeu_ps(y + l + j * 8, result);
        }
    }
}

// Dequantize a block of weight
static void dequantize_block_q4_zp_no_offset_v2(const uint8_t *int4_w, float *y, float scale, int block_size) {
    float zp = 8.0f;

    const __m256 d_v = _mm256_broadcast_ss(&scale);
    const __m256 d_zp = _mm256_broadcast_ss(&zp);

    const uint8_t *pp = int4_w;

    for (int l = 0; l < block_size; l += 32) {
        // Load 32x4-bit integers into 32x8-bit integers
        __m256i vx8 = bytes_from_nibbles_32(pp + l / 2);

        // Convert to 16-bit int
        __m128i lo_128 = _mm256_extracti128_si256(vx8, 0);
        __m128i hi_128 = _mm256_extracti128_si256(vx8, 1);
        const __m256i vx16_lo = _mm256_cvtepi8_epi16(lo_128);
        const __m256i vx16_hi = _mm256_cvtepi8_epi16(hi_128);

        // Convert to 32-bit int -> float 32
        __m128i lo_128_0 = _mm256_extracti128_si256(vx16_lo, 0);
        __m128i lo_128_1 = _mm256_extracti128_si256(vx16_lo, 1);
        __m128i hi_128_1 = _mm256_extracti128_si256(vx16_hi, 1);
        __m128i hi_128_0 = _mm256_extracti128_si256(vx16_hi, 0);
        __m256i lo_256_0 = _mm256_cvtepi16_epi32(lo_128_0);
        __m256i lo_256_1 = _mm256_cvtepi16_epi32(lo_128_1);
        __m256i hi_256_0 = _mm256_cvtepi16_epi32(hi_128_0);
        __m256i hi_256_1 = _mm256_cvtepi16_epi32(hi_128_1);
        __m256 lo_fp_0 = _mm256_cvtepi32_ps(lo_256_0);
        __m256 lo_fp_1 = _mm256_cvtepi32_ps(lo_256_1);
        __m256 hi_fp_0 = _mm256_cvtepi32_ps(hi_256_0);
        __m256 hi_fp_1 = _mm256_cvtepi32_ps(hi_256_1);
        __m256 lo_zp_0 = _mm256_sub_ps(lo_fp_0, d_zp);
        __m256 lo_zp_1 = _mm256_sub_ps(lo_fp_1, d_zp);
        __m256 hi_zp_0 = _mm256_sub_ps(hi_fp_0, d_zp);
        __m256 hi_zp_1 = _mm256_sub_ps(hi_fp_1, d_zp);
        _mm256_storeu_ps(y + l + 0 * 8, _mm256_mul_ps(lo_zp_0, d_v));
        _mm256_storeu_ps(y + l + 1 * 8, _mm256_mul_ps(lo_zp_1, d_v));
        _mm256_storeu_ps(y + l + 2 * 8, _mm256_mul_ps(hi_zp_0, d_v));
        _mm256_storeu_ps(y + l + 3 * 8, _mm256_mul_ps(hi_zp_1, d_v));
    }
}

// Dequantize a block of weight
static inline void dequantize_madd_block_q4_zp_no_offset(const uint8_t *int4_w, float scale, int block_size,
                                                         __m256 *x_ptr, __m256 &acc) {
    float zp = 8.0f;

    const __m256 d_v = _mm256_broadcast_ss(&scale);
    const __m256 d_zp = _mm256_broadcast_ss(&zp);

    const uint8_t *pp = int4_w;

    for (int l = 0; l < block_size; l += 32) {
        // Load 32x4-bit integers into 32x8-bit integers
        __m256i vx8 = bytes_from_nibbles_32(pp + l / 2);

        // Convert to 16-bit int
        __m128i lo_128 = _mm256_extracti128_si256(vx8, 0);
        __m128i hi_128 = _mm256_extracti128_si256(vx8, 1);
        const __m256i vx16_lo = _mm256_cvtepi8_epi16(lo_128);
        const __m256i vx16_hi = _mm256_cvtepi8_epi16(hi_128);

        // Convert to 32-bit int -> float 32
        __m128i lo_128_0 = _mm256_extracti128_si256(vx16_lo, 0);
        __m128i lo_128_1 = _mm256_extracti128_si256(vx16_lo, 1);
        __m128i hi_128_1 = _mm256_extracti128_si256(vx16_hi, 1);
        __m128i hi_128_0 = _mm256_extracti128_si256(vx16_hi, 0);
        __m256i lo_256_0 = _mm256_cvtepi16_epi32(lo_128_0);
        __m256i lo_256_1 = _mm256_cvtepi16_epi32(lo_128_1);
        __m256i hi_256_0 = _mm256_cvtepi16_epi32(hi_128_0);
        __m256i hi_256_1 = _mm256_cvtepi16_epi32(hi_128_1);
        __m256 lo_fp_0 = _mm256_cvtepi32_ps(lo_256_0);
        __m256 lo_fp_1 = _mm256_cvtepi32_ps(lo_256_1);
        __m256 hi_fp_0 = _mm256_cvtepi32_ps(hi_256_0);
        __m256 hi_fp_1 = _mm256_cvtepi32_ps(hi_256_1);
        __m256 lo_zp_0 = _mm256_sub_ps(lo_fp_0, d_zp);
        __m256 lo_zp_1 = _mm256_sub_ps(lo_fp_1, d_zp);
        __m256 hi_zp_0 = _mm256_sub_ps(hi_fp_0, d_zp);
        __m256 hi_zp_1 = _mm256_sub_ps(hi_fp_1, d_zp);

        acc = _mm256_fmadd_ps(*x_ptr++, _mm256_mul_ps(lo_zp_0, d_v), acc);
        acc = _mm256_fmadd_ps(*x_ptr++, _mm256_mul_ps(lo_zp_1, d_v), acc);
        acc = _mm256_fmadd_ps(*x_ptr++, _mm256_mul_ps(hi_zp_0, d_v), acc);
        acc = _mm256_fmadd_ps(*x_ptr++, _mm256_mul_ps(hi_zp_1, d_v), acc);
    }
}

static inline void dequantize_madd_block_q4_zp_no_offset_unroll2(const uint8_t *int4_w, const uint8_t *int4_w2,
                                                                 float scale, int block_size, __m256 *x_ptr,
                                                                 __m256 &acc0, __m256 &acc1) {
    float zp = 8.0f;

    const __m256 d_v = _mm256_broadcast_ss(&scale);
    const __m256 d_zp = _mm256_broadcast_ss(&zp);

    const uint8_t *pp = int4_w;
    const uint8_t *pp2 = int4_w2;

    for (int l = 0; l < block_size; l += 32) {
        // Load 32x4-bit integers into 32x8-bit integers
        __m256i vx8 = bytes_from_nibbles_32(pp + l / 2);

        // Convert to 16-bit int
        __m128i lo_128 = _mm256_extracti128_si256(vx8, 0);
        __m128i hi_128 = _mm256_extracti128_si256(vx8, 1);
        const __m256i vx16_lo = _mm256_cvtepi8_epi16(lo_128);
        const __m256i vx16_hi = _mm256_cvtepi8_epi16(hi_128);

        // Convert to 32-bit int -> float 32
        __m128i lo_128_0 = _mm256_extracti128_si256(vx16_lo, 0);
        __m128i lo_128_1 = _mm256_extracti128_si256(vx16_lo, 1);
        __m128i hi_128_1 = _mm256_extracti128_si256(vx16_hi, 1);
        __m128i hi_128_0 = _mm256_extracti128_si256(vx16_hi, 0);

        __m256i lo_256_0 = _mm256_cvtepi16_epi32(lo_128_0);
        __m256i lo_256_1 = _mm256_cvtepi16_epi32(lo_128_1);
        __m256i hi_256_0 = _mm256_cvtepi16_epi32(hi_128_0);
        __m256i hi_256_1 = _mm256_cvtepi16_epi32(hi_128_1);
        __m256 lo_fp_0 = _mm256_cvtepi32_ps(lo_256_0);
        __m256 lo_fp_1 = _mm256_cvtepi32_ps(lo_256_1);
        __m256 hi_fp_0 = _mm256_cvtepi32_ps(hi_256_0);
        __m256 hi_fp_1 = _mm256_cvtepi32_ps(hi_256_1);

        __m256 lo_zp_0 = _mm256_sub_ps(lo_fp_0, d_zp);
        __m256 lo_zp_1 = _mm256_sub_ps(lo_fp_1, d_zp);
        __m256 hi_zp_0 = _mm256_sub_ps(hi_fp_0, d_zp);
        __m256 hi_zp_1 = _mm256_sub_ps(hi_fp_1, d_zp);

        acc0 = _mm256_fmadd_ps(x_ptr[0], _mm256_mul_ps(lo_zp_0, d_v), acc0);
        acc0 = _mm256_fmadd_ps(x_ptr[1], _mm256_mul_ps(lo_zp_1, d_v), acc0);
        acc0 = _mm256_fmadd_ps(x_ptr[2], _mm256_mul_ps(hi_zp_0, d_v), acc0);
        acc0 = _mm256_fmadd_ps(x_ptr[3], _mm256_mul_ps(hi_zp_1, d_v), acc0);

        __m256i vx82 = bytes_from_nibbles_32(pp2 + l / 2);
        __m128i lo2_128 = _mm256_extracti128_si256(vx82, 0);
        __m128i hi2_128 = _mm256_extracti128_si256(vx82, 1);
        const __m256i vx16_lo2 = _mm256_cvtepi8_epi16(lo2_128);
        const __m256i vx16_hi2 = _mm256_cvtepi8_epi16(hi2_128);

        __m128i lo2_128_0 = _mm256_extracti128_si256(vx16_lo2, 0);
        __m128i lo2_128_1 = _mm256_extracti128_si256(vx16_lo2, 1);
        __m128i hi2_128_1 = _mm256_extracti128_si256(vx16_hi2, 1);
        __m128i hi2_128_0 = _mm256_extracti128_si256(vx16_hi2, 0);

        __m256i lo2_256_0 = _mm256_cvtepi16_epi32(lo2_128_0);
        __m256i lo2_256_1 = _mm256_cvtepi16_epi32(lo2_128_1);
        __m256i hi2_256_0 = _mm256_cvtepi16_epi32(hi2_128_0);
        __m256i hi2_256_1 = _mm256_cvtepi16_epi32(hi2_128_1);

        __m256 lo2_fp_0 = _mm256_cvtepi32_ps(lo2_256_0);
        __m256 lo2_fp_1 = _mm256_cvtepi32_ps(lo2_256_1);
        __m256 hi2_fp_0 = _mm256_cvtepi32_ps(hi2_256_0);
        __m256 hi2_fp_1 = _mm256_cvtepi32_ps(hi2_256_1);

        __m256 lo2_zp_0 = _mm256_sub_ps(lo2_fp_0, d_zp);
        __m256 lo2_zp_1 = _mm256_sub_ps(lo2_fp_1, d_zp);
        __m256 hi2_zp_0 = _mm256_sub_ps(hi2_fp_0, d_zp);
        __m256 hi2_zp_1 = _mm256_sub_ps(hi2_fp_1, d_zp);

        acc1 = _mm256_fmadd_ps(x_ptr[0], _mm256_mul_ps(lo2_zp_0, d_v), acc1);
        acc1 = _mm256_fmadd_ps(x_ptr[1], _mm256_mul_ps(lo2_zp_1, d_v), acc1);
        acc1 = _mm256_fmadd_ps(x_ptr[2], _mm256_mul_ps(hi2_zp_0, d_v), acc1);
        acc1 = _mm256_fmadd_ps(x_ptr[3], _mm256_mul_ps(hi2_zp_1, d_v), acc1);
    }
}

static inline void dequantize_madd_block_q4_zp_no_offset_unroll2(const uint8_t *int4_w, const uint8_t *int4_w2,
                                                                 int block_size, __m256 *x_ptr, __m256 &acc0,
                                                                 __m256 &acc1) {
    float zp = 8.0f;

    const __m256 d_zp = _mm256_broadcast_ss(&zp);

    const uint8_t *pp = int4_w;
    const uint8_t *pp2 = int4_w2;

    for (int l = 0; l < block_size; l += 32) {
        // Load 32x4-bit integers into 32x8-bit integers
        __m256i vx8 = bytes_from_nibbles_32(pp + l / 2);

        // Convert to 16-bit int
        __m128i lo_128 = _mm256_extracti128_si256(vx8, 0);
        __m128i hi_128 = _mm256_extracti128_si256(vx8, 1);
        const __m256i vx16_lo = _mm256_cvtepi8_epi16(lo_128);
        const __m256i vx16_hi = _mm256_cvtepi8_epi16(hi_128);

        // Convert to 32-bit int -> float 32
        __m128i lo_128_0 = _mm256_extracti128_si256(vx16_lo, 0);
        __m128i lo_128_1 = _mm256_extracti128_si256(vx16_lo, 1);
        __m128i hi_128_1 = _mm256_extracti128_si256(vx16_hi, 1);
        __m128i hi_128_0 = _mm256_extracti128_si256(vx16_hi, 0);

        __m256i lo_256_0 = _mm256_cvtepi16_epi32(lo_128_0);
        __m256i lo_256_1 = _mm256_cvtepi16_epi32(lo_128_1);
        __m256i hi_256_0 = _mm256_cvtepi16_epi32(hi_128_0);
        __m256i hi_256_1 = _mm256_cvtepi16_epi32(hi_128_1);
        __m256 lo_fp_0 = _mm256_cvtepi32_ps(lo_256_0);
        __m256 lo_fp_1 = _mm256_cvtepi32_ps(lo_256_1);
        __m256 hi_fp_0 = _mm256_cvtepi32_ps(hi_256_0);
        __m256 hi_fp_1 = _mm256_cvtepi32_ps(hi_256_1);

        __m256 lo_zp_0 = _mm256_sub_ps(lo_fp_0, d_zp);
        __m256 lo_zp_1 = _mm256_sub_ps(lo_fp_1, d_zp);
        __m256 hi_zp_0 = _mm256_sub_ps(hi_fp_0, d_zp);
        __m256 hi_zp_1 = _mm256_sub_ps(hi_fp_1, d_zp);

        acc0 = _mm256_fmadd_ps(x_ptr[0], lo_zp_0, acc0);
        acc0 = _mm256_fmadd_ps(x_ptr[1], lo_zp_1, acc0);
        acc0 = _mm256_fmadd_ps(x_ptr[2], hi_zp_0, acc0);
        acc0 = _mm256_fmadd_ps(x_ptr[3], hi_zp_1, acc0);

        __m256i vx82 = bytes_from_nibbles_32(pp2 + l / 2);
        __m128i lo2_128 = _mm256_extracti128_si256(vx82, 0);
        __m128i hi2_128 = _mm256_extracti128_si256(vx82, 1);
        const __m256i vx16_lo2 = _mm256_cvtepi8_epi16(lo2_128);
        const __m256i vx16_hi2 = _mm256_cvtepi8_epi16(hi2_128);

        __m128i lo2_128_0 = _mm256_extracti128_si256(vx16_lo2, 0);
        __m128i lo2_128_1 = _mm256_extracti128_si256(vx16_lo2, 1);
        __m128i hi2_128_1 = _mm256_extracti128_si256(vx16_hi2, 1);
        __m128i hi2_128_0 = _mm256_extracti128_si256(vx16_hi2, 0);

        __m256i lo2_256_0 = _mm256_cvtepi16_epi32(lo2_128_0);
        __m256i lo2_256_1 = _mm256_cvtepi16_epi32(lo2_128_1);
        __m256i hi2_256_0 = _mm256_cvtepi16_epi32(hi2_128_0);
        __m256i hi2_256_1 = _mm256_cvtepi16_epi32(hi2_128_1);

        __m256 lo2_fp_0 = _mm256_cvtepi32_ps(lo2_256_0);
        __m256 lo2_fp_1 = _mm256_cvtepi32_ps(lo2_256_1);
        __m256 hi2_fp_0 = _mm256_cvtepi32_ps(hi2_256_0);
        __m256 hi2_fp_1 = _mm256_cvtepi32_ps(hi2_256_1);

        __m256 lo2_zp_0 = _mm256_sub_ps(lo2_fp_0, d_zp);
        __m256 lo2_zp_1 = _mm256_sub_ps(lo2_fp_1, d_zp);
        __m256 hi2_zp_0 = _mm256_sub_ps(hi2_fp_0, d_zp);
        __m256 hi2_zp_1 = _mm256_sub_ps(hi2_fp_1, d_zp);

        acc1 = _mm256_fmadd_ps(x_ptr[0], lo2_zp_0, acc1);
        acc1 = _mm256_fmadd_ps(x_ptr[1], lo2_zp_1, acc1);
        acc1 = _mm256_fmadd_ps(x_ptr[2], hi2_zp_0, acc1);
        acc1 = _mm256_fmadd_ps(x_ptr[3], hi2_zp_1, acc1);
    }
}

// Dequantize two block of weight
static void dequantize_two_block_q4_zp_no_offset(const uint8_t *int4_w, float *y, float scale, const uint8_t *int4_w2,
                                                 float *y2, float scale2, int block_size) {
    float z = 8.0f;

    const __m256 d_v = _mm256_broadcast_ss(&scale);
    const __m256 d_v2 = _mm256_broadcast_ss(&scale2);
    const __m256 d_zp = _mm256_broadcast_ss(&z);

    const uint8_t *pp = int4_w;
    const uint8_t *pp2 = int4_w2;

    for (int l = 0; l < block_size; l += 32) {
        // Load 32x4-bit integers into 32x8-bit integers
        __m256i vx8 = bytes_from_nibbles_32(pp + l / 2);
        __m256i vx28 = bytes_from_nibbles_32(pp2 + l / 2);

        // Convert to 16-bit int
        __m128i lo_128 = _mm256_extracti128_si256(vx8, 0);
        __m128i lo2_128 = _mm256_extracti128_si256(vx28, 0);
        const __m256i vx16_lo = _mm256_cvtepi8_epi16(lo_128);
        const __m256i vx16_lo2 = _mm256_cvtepi8_epi16(lo2_128);

        __m128i hi_128 = _mm256_extracti128_si256(vx8, 1);
        __m128i hi2_128 = _mm256_extracti128_si256(vx28, 1);
        const __m256i vx16_hi = _mm256_cvtepi8_epi16(hi_128);
        const __m256i vx16_hi2 = _mm256_cvtepi8_epi16(hi2_128);

        // Convert to 32-bit int -> float 32
        __m128i lo_128_0 = _mm256_extracti128_si256(vx16_lo, 0);
        __m128i lo_128_1 = _mm256_extracti128_si256(vx16_lo, 1);
        __m256i lo_256_0 = _mm256_cvtepi16_epi32(lo_128_0);
        __m256i lo_256_1 = _mm256_cvtepi16_epi32(lo_128_1);
        __m256 lo_fp_0 = _mm256_cvtepi32_ps(lo_256_0);
        __m256 lo_fp_1 = _mm256_cvtepi32_ps(lo_256_1);
        __m256 lo_zp_0 = _mm256_sub_ps(lo_fp_0, d_zp);
        __m256 lo_zp_1 = _mm256_sub_ps(lo_fp_1, d_zp);
        _mm256_storeu_ps(y + l + 0 * 8, _mm256_mul_ps(lo_zp_0, d_v));
        _mm256_storeu_ps(y + l + 1 * 8, _mm256_mul_ps(lo_zp_1, d_v));

        __m128i hi_128_0 = _mm256_extracti128_si256(vx16_hi, 0);
        __m128i hi_128_1 = _mm256_extracti128_si256(vx16_hi, 1);
        __m256i hi_256_0 = _mm256_cvtepi16_epi32(hi_128_0);
        __m256i hi_256_1 = _mm256_cvtepi16_epi32(hi_128_1);
        __m256 hi_fp_0 = _mm256_cvtepi32_ps(hi_256_0);
        __m256 hi_fp_1 = _mm256_cvtepi32_ps(hi_256_1);
        __m256 hi_zp_0 = _mm256_sub_ps(hi_fp_0, d_zp);
        __m256 hi_zp_1 = _mm256_sub_ps(hi_fp_1, d_zp);
        _mm256_storeu_ps(y + l + 2 * 8, _mm256_mul_ps(hi_zp_0, d_v));
        _mm256_storeu_ps(y + l + 3 * 8, _mm256_mul_ps(hi_zp_1, d_v));

        // Convert to 32-bit int -> block 2
        __m128i lo2_128_0 = _mm256_extracti128_si256(vx16_lo2, 0);
        __m128i lo2_128_1 = _mm256_extracti128_si256(vx16_lo2, 1);
        __m256i lo2_256_0 = _mm256_cvtepi16_epi32(lo2_128_0);
        __m256i lo2_256_1 = _mm256_cvtepi16_epi32(lo2_128_1);
        __m256 lo2_fp_0 = _mm256_cvtepi32_ps(lo2_256_0);
        __m256 lo2_fp_1 = _mm256_cvtepi32_ps(lo2_256_1);
        __m256 lo2_zp_0 = _mm256_sub_ps(lo2_fp_0, d_zp);
        __m256 lo2_zp_1 = _mm256_sub_ps(lo2_fp_1, d_zp);
        _mm256_storeu_ps(y2 + l + 0 * 8, _mm256_mul_ps(lo2_zp_0, d_v2));
        _mm256_storeu_ps(y2 + l + 1 * 8, _mm256_mul_ps(lo2_zp_1, d_v2));

        __m128i hi2_128_0 = _mm256_extracti128_si256(vx16_hi2, 0);
        __m128i hi2_128_1 = _mm256_extracti128_si256(vx16_hi2, 1);
        __m256i hi2_256_0 = _mm256_cvtepi16_epi32(hi2_128_0);
        __m256i hi2_256_1 = _mm256_cvtepi16_epi32(hi2_128_1);
        __m256 hi2_fp_0 = _mm256_cvtepi32_ps(hi2_256_0);
        __m256 hi2_fp_1 = _mm256_cvtepi32_ps(hi2_256_1);
        __m256 hi2_zp_0 = _mm256_sub_ps(hi2_fp_0, d_zp);
        __m256 hi2_zp_1 = _mm256_sub_ps(hi2_fp_1, d_zp);
        _mm256_storeu_ps(y2 + l + 2 * 8, _mm256_mul_ps(hi2_zp_0, d_v2));
        _mm256_storeu_ps(y2 + l + 3 * 8, _mm256_mul_ps(hi2_zp_1, d_v2));
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
            for (k = 0; k < B->row * 2; k += block_size) {
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
            for (k = 0; k < B->row * 2; k += block_size) {
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

static void *fast_zp_no_offset_over_column_func_v1(void *args) {
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
            for (k = 0; k < B->row * 2; k += block_size) {
                float s = scale[j * (B->row / 16) + k / 32];  // /16:B->row is packed 4bits
                // float o = offset[j * (B->column / 16) + k / 32];
                // float zp = zero_point(0, j, k/32);
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->row + k / 2];
                __m256 *x_ptr = (__m256 *)&A->data_ptr[i * A->column + k];
                __m256 *w_ptr = (__m256 *)&weight_block;
                dequantize_block_q4_zp_no_offset_v2(weight_32_int4, weight_block, s, block_size);

                // assume block_size == 32 (8 x 4 float)
                // acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(x_ptr[0], w_ptr[0]));
                // acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(x_ptr[1], w_ptr[1]));
                // acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(x_ptr[2], w_ptr[2]));
                // acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(x_ptr[3], w_ptr[3]));
                acc0 = _mm256_fmadd_ps(x_ptr[0], w_ptr[0], acc0);
                acc0 = _mm256_fmadd_ps(x_ptr[1], w_ptr[1], acc0);
                acc0 = _mm256_fmadd_ps(x_ptr[2], w_ptr[2], acc0);
                acc0 = _mm256_fmadd_ps(x_ptr[3], w_ptr[3], acc0);
            }
            float *ptr = (float *)&acc0;
            C->data_ptr[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
        }
    }
    return NULL;
}

static void *fast_zp_no_offset_over_column_func_v2(void *args) {
    int i, j, k;
    struct int4_thread_args *mat_args = (struct int4_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    float weight_block[block_size];
    float weight2_block[block_size];

    for (i = 0; i < C->row; i++) {
        for (j = mat_args->start_j; j < mat_args->end_j; j += 2) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (k = 0; k < B->row * 2; k += block_size) {
                float s = scale[j * (B->row / 16) + k / 32];         // /16:B->row is packed 4bits
                float s2 = scale[(j + 1) * (B->row / 16) + k / 32];  // /16:B->row is packed 4bits
                // float o = offset[j * (B->column / 16) + k / 32];
                // float zp = zero_point(0, j, k/32);
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->row + k / 2];
                uint8_t *weight2_32_int4 = &B->int4_data_ptr[(j + 1) * B->row + k / 2];
                __m256 *x_ptr = (__m256 *)&A->data_ptr[i * A->column + k];
                dequantize_two_block_q4_zp_no_offset(weight_32_int4, weight_block, s, weight2_32_int4, weight2_block,
                                                     s2, block_size);

                __m256 *w_ptr = (__m256 *)&weight_block;
                __m256 *w2_ptr = (__m256 *)&weight2_block;
                // assume block_size == 32 (8 x 4 float)
                acc0 = _mm256_fmadd_ps(x_ptr[0], w_ptr[0], acc0);
                acc1 = _mm256_fmadd_ps(x_ptr[0], w2_ptr[0], acc1);
                acc0 = _mm256_fmadd_ps(x_ptr[1], w_ptr[1], acc0);
                acc1 = _mm256_fmadd_ps(x_ptr[1], w2_ptr[1], acc1);
                acc0 = _mm256_fmadd_ps(x_ptr[2], w_ptr[2], acc0);
                acc1 = _mm256_fmadd_ps(x_ptr[2], w2_ptr[2], acc1);
                acc0 = _mm256_fmadd_ps(x_ptr[3], w_ptr[3], acc0);
                acc1 = _mm256_fmadd_ps(x_ptr[3], w2_ptr[3], acc1);
                // acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(x_ptr[0], w_ptr[0]));
                // acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(x_ptr[0], w2_ptr[0]));
                // acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(x_ptr[1], w_ptr[1]));
                // acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(x_ptr[1], w2_ptr[1]));
                // acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(x_ptr[2], w_ptr[2]));
                // acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(x_ptr[2], w2_ptr[2]));
                // acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(x_ptr[3], w_ptr[3]));
                // acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(x_ptr[3], w2_ptr[3]));
            }
            float *ptr = (float *)&acc0;
            C->data_ptr[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
            ptr = (float *)&acc1;
            C->data_ptr[i * C->column + j + 1] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
        }
    }
    return NULL;
}

static void *fast_zp_no_offset_over_column_func_v3(void *args) {
    int i, j, k;
    struct int4_thread_args *mat_args = (struct int4_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    for (i = 0; i < C->row; i++) {
        for (j = mat_args->start_j; j < mat_args->end_j; j++) {
            __m256 acc0 = _mm256_setzero_ps();
            for (k = 0; k < B->row * 2; k += block_size) {
                float s = scale[j * (B->row / 16) + k / 32];  // /16:B->row is packed 4bits
                // float o = offset[j * (B->column / 16) + k / 32];
                // float zp = zero_point(0, j, k/32);
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->row + k / 2];
                __m256 *x_ptr = (__m256 *)&A->data_ptr[i * A->column + k];
                dequantize_madd_block_q4_zp_no_offset(weight_32_int4, s, block_size, x_ptr, acc0);
            }
            float *ptr = (float *)&acc0;
            C->data_ptr[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
        }
    }
    return NULL;
}

static void *fast_zp_no_offset_over_column_func_v4(void *args) {
    int i, j, k;
    struct int4_thread_args *mat_args = (struct int4_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    for (i = 0; i < C->row; i++) {
        for (j = mat_args->start_j; j < mat_args->end_j; j++) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (k = 0; k < B->row * 2; k += block_size) {
                float s = scale[j * (B->row / 16) + k / 32];  // /16:B->row is packed 4bits
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->row + k / 2];
                uint8_t *weight2_32_int4 = &B->int4_data_ptr[(j + 1) * B->row + k / 2];
                __m256 *x_ptr = (__m256 *)&A->data_ptr[i * A->column + k];
                dequantize_madd_block_q4_zp_no_offset_unroll2(weight_32_int4, weight2_32_int4, s, block_size, x_ptr,
                                                              acc0, acc1);
            }
            float *ptr = (float *)&acc0;
            C->data_ptr[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
        }
    }
    return NULL;
}

static void *fast_zp_no_offset_over_column_func_v5(void *args) {
    int i, j, k;
    struct int4_thread_args *mat_args = (struct int4_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    for (i = 0; i < C->row; i++) {
        for (j = mat_args->start_j; j < mat_args->end_j; j += 2) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (k = 0; k < B->row * 2; k += block_size) {
                // float s = scale[j * (B->row / 16) + k / 32];  // /16:B->row is packed 4bits
                __m256 part_sum0 = _mm256_setzero_ps();
                __m256 part_sum1 = _mm256_setzero_ps();
                uint8_t *weight_32_int4 = &B->int4_data_ptr[j * B->row + k / 2];
                uint8_t *weight2_32_int4 = &B->int4_data_ptr[(j + 1) * B->row + k / 2];
                __m256 *x_ptr = (__m256 *)&A->data_ptr[i * A->column + k];
                dequantize_madd_block_q4_zp_no_offset_unroll2(weight_32_int4, weight2_32_int4, block_size, x_ptr,
                                                              part_sum0, part_sum1);

                const __m256 s_v = _mm256_broadcast_ss(&scale[j * (B->row / 16) + k / 32]);
                const __m256 s2_v = _mm256_broadcast_ss(&scale[(j + 1) * (B->row / 16) + k / 32]);
                part_sum0 = _mm256_mul_ps(part_sum0, s_v);
                part_sum1 = _mm256_mul_ps(part_sum1, s2_v);

                acc0 = _mm256_add_ps(part_sum0, acc0);
                acc1 = _mm256_add_ps(part_sum1, acc1);
            }
            float *ptr = (float *)&acc0;
            C->data_ptr[i * C->column + j] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
            ptr = (float *)&acc1;
            C->data_ptr[i * C->column + j + 1] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
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

void MatmulOperator::mat_mul_accelerator_int4_fast_no_offset(const struct matmul_params *params) {
    const int num_thread = params->opt_params.num_thread;
    int i, j, k;
    pthread_t thread_pool[num_thread];
    struct int4_thread_args threads_args[num_thread];
    assert(params->block_size == 32);                    // support block size 32 for now
    assert((params->C.column % (num_thread * 2)) == 0);  // support block size 32 for now

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_j = j * (params->C.column / num_thread);
        threads_args[j].end_j = (j + 1) * (params->C.column / num_thread);
        threads_args[j].params = params;
        pthread_create(&thread_pool[j], NULL, fast_zp_no_offset_over_column_func_v5, &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) pthread_join(thread_pool[j], NULL);
};
}  // namespace matmul
