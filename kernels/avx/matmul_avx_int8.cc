#include <immintrin.h>  // AVX instrintics

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "../matmul.h"

inline void assign_8int32(int *ptr, int &acc) {
    acc = (ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7]);
}

inline void clamp_max_min(int &value, const int8_t min, const int8_t max) {}

namespace matmul {
void dump_64x8_signed(__m256i &target, char *title) {
    int8_t *ptr = (int8_t *)&target;

    printf("%s:", title);
    for (int i = 0; i < 64; i++) {
        printf("%3d, ", *ptr++);
    }
    printf("\n");
}

void dump_64x8_unsigned(__m256i &target, char *title) {
    uint8_t *ptr = (uint8_t *)&target;

    printf("%s:", title);
    for (int i = 0; i < 64; i++) {
        printf("%3d, ", *ptr++);
    }
    printf("\n");
}

void dump_16x16_signed(__m256i &target, char *title) {
    int16_t *ptr = (int16_t *)&target;

    printf("%s:", title);
    for (int i = 0; i < 16; i++) {
        printf("%d, ", *ptr++);
    }
    printf("\n");
}

// element-wise multiply two vectors of 64 8-bit integers and return the accumulate 32-bit result
// We need to assume int8 is in the range of 127 <-> - 127, otherwise, we will expect overflow in some case
// e,g., a[i] = b[i] = -128 -> a[i] * b[i] = 32768 which is not in the range of int16_t(-32768, 32767)
__m256i zero_vec = _mm256_setzero_si256();
__m256i multiply_signed_int8(__m256i &a, __m256i &b, __m256i &a2, __m256i &b2) {
    __m256i a_sign_mask = _mm256_cmpgt_epi8(zero_vec, a);    // set 0xFF if zero_vec[i] > a[i]
    __m256i b_sign_mask = _mm256_cmpgt_epi8(zero_vec, b);    // set 0xFF if zero_vec[i] > a[i]
    __m256i a2_sign_mask = _mm256_cmpgt_epi8(zero_vec, a2);  // set 0xFF if zero_vec[i] > a[i]
    __m256i b2_sign_mask = _mm256_cmpgt_epi8(zero_vec, b2);  // set 0xFF if zero_vec[i] > a[i]

    // Compute the two's complement of a, put it here for higher throughput with good instruction dep.
    __m256i b_abs = _mm256_abs_epi8(b);
    __m256i b2_abs = _mm256_abs_epi8(b2);
    __m256i a_abs = _mm256_abs_epi8(a);
    __m256i a2_abs = _mm256_abs_epi8(a2);
    __m256i b_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), b_abs);
    __m256i b2_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), b2_abs);

    // Manipulate the `sign` of B to represent the sign of the 16 bit result
    __m256i sign_mask_a_sub_b = _mm256_sub_epi8(a_sign_mask, b_sign_mask);
    __m256i sign_mask_a2_sub_b2 = _mm256_sub_epi8(a2_sign_mask, b2_sign_mask);
    __m256i sign_mask =
        _mm256_cmpeq_epi8(sign_mask_a_sub_b, zero_vec);  // sign_mask[i] if a[i] and b[i] have different sign bits
    __m256i sign_mask2 = _mm256_cmpeq_epi8(sign_mask_a2_sub_b2, zero_vec);
    __m256i corrected_b = _mm256_blendv_epi8(b_negated, b_abs, sign_mask);
    __m256i corrected_b2 = _mm256_blendv_epi8(b2_negated, b2_abs, sign_mask2);

    // Multiply the absolute values of a_abs (unsigned 8-bit integers) and corrected_b (signed 8-bit integers)
    __m256i product_16x16 = _mm256_maddubs_epi16(a_abs, corrected_b);
    __m256i product_16x16_2 = _mm256_maddubs_epi16(a2_abs, corrected_b2);

    // Sign extend the 16-bit integers in vector to 32-bit integers
    __m256i a_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16, 0));
    __m256i b_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_2, 0));
    __m256i a_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16, 1));
    __m256i b_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_2, 1));

    // Element-wise add the 32-bit integer vectors
    __m256i sum1 = _mm256_add_epi32(a_ext1, b_ext1);
    __m256i sum2 = _mm256_add_epi32(a_ext2, b_ext2);

    __m256i sum_product_8x32 = _mm256_add_epi32(sum1, sum2);

    return sum_product_8x32;
}

static inline __m256i multiply_signed_int8_32epi(__m256i &a, __m256i &b) {
    __m256i a_sign_mask = _mm256_cmpgt_epi8(zero_vec, a);  // set 0xFF if zero_vec[i] > a[i]
    __m256i b_sign_mask = _mm256_cmpgt_epi8(zero_vec, b);  // set 0xFF if zero_vec[i] > a[i]

    // Compute the two's complement of a, put it here for higher throughput with good instruction dep.
    __m256i b_abs = _mm256_abs_epi8(b);
    __m256i a_abs = _mm256_abs_epi8(a);
    __m256i b_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), b_abs);

    // Manipulate the `sign` of B to represent the sign of the 16 bit result
    __m256i sign_mask_a_sub_b = _mm256_sub_epi8(a_sign_mask, b_sign_mask);
    __m256i sign_mask =
        _mm256_cmpeq_epi8(sign_mask_a_sub_b, zero_vec);  // sign_mask[i] if a[i] and b[i] have different sign bits
    __m256i corrected_b = _mm256_blendv_epi8(b_negated, b_abs, sign_mask);

    // Multiply the absolute values of a_abs (unsigned 8-bit integers) and corrected_b (signed 8-bit integers)
    __m256i product_16x16 = _mm256_maddubs_epi16(a_abs, corrected_b);

    // Sign extend the 16-bit integers in vector to 32-bit integers
    __m256i a_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16, 0));
    __m256i a_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16, 1));

    // Element-wise add the 32-bit integer vectors
    __m256i sum1 = _mm256_add_epi32(a_ext1, a_ext2);

    return sum1;
}

// ([a, a2], [c, c2])  *  ([b, b2], [d, d2])
//  acc0 = a * b + a2 * b2, acc2 = a * d + a2 * d2, acc3 = c * b + c * b2, acc4 = c * d + c2 * d2
void multiply_signed_int8_2x2(__m256i &a, __m256i &b, __m256i &a2, __m256i &b2, __m256i &c, __m256i &c2, __m256i &d,
                              __m256i &d2, __m256i &acc0, __m256i &acc1, __m256i &acc2, __m256i &acc3) {
    __m256i a_sign_mask = _mm256_cmpgt_epi8(zero_vec, a);  // set 0xFF if zero_vec[i] > a[i]
    __m256i b_sign_mask = _mm256_cmpgt_epi8(zero_vec, b);  // set 0xFF if zero_vec[i] > a[i]
    __m256i a2_sign_mask = _mm256_cmpgt_epi8(zero_vec, a2);
    __m256i b2_sign_mask = _mm256_cmpgt_epi8(zero_vec, b2);
    __m256i c_sign_mask = _mm256_cmpgt_epi8(zero_vec, c);
    __m256i d_sign_mask = _mm256_cmpgt_epi8(zero_vec, d);
    __m256i c2_sign_mask = _mm256_cmpgt_epi8(zero_vec, c2);
    __m256i d2_sign_mask = _mm256_cmpgt_epi8(zero_vec, d2);

    // Compute the two's complement of a, put it here for higher throughput with good instruction dep.
    __m256i b_abs = _mm256_abs_epi8(b);
    __m256i b2_abs = _mm256_abs_epi8(b2);
    __m256i a_abs = _mm256_abs_epi8(a);
    __m256i a2_abs = _mm256_abs_epi8(a2);
    __m256i d_abs = _mm256_abs_epi8(d);
    __m256i d2_abs = _mm256_abs_epi8(d2);
    __m256i c_abs = _mm256_abs_epi8(c);
    __m256i c2_abs = _mm256_abs_epi8(c2);
    __m256i b_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), b_abs);
    __m256i b2_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), b2_abs);
    __m256i d_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), d_abs);
    __m256i d2_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), d2_abs);

    // Manipulate the `sign` of B to represent the sign of the 16 bit result
    __m256i sign_mask_a_sub_b = _mm256_sub_epi8(a_sign_mask, b_sign_mask);
    __m256i sign_mask_a_sub_d = _mm256_sub_epi8(a_sign_mask, d_sign_mask);
    __m256i sign_mask_a2_sub_b2 = _mm256_sub_epi8(a2_sign_mask, b2_sign_mask);
    __m256i sign_mask_a2_sub_d2 = _mm256_sub_epi8(a2_sign_mask, d2_sign_mask);
    __m256i sign_mask_c_sub_b = _mm256_sub_epi8(c_sign_mask, b_sign_mask);
    __m256i sign_mask_c_sub_d = _mm256_sub_epi8(c_sign_mask, d_sign_mask);
    __m256i sign_mask_c2_sub_b2 = _mm256_sub_epi8(c2_sign_mask, b2_sign_mask);
    __m256i sign_mask_c2_sub_d2 = _mm256_sub_epi8(c2_sign_mask, d2_sign_mask);

    // sign_mask[i] if a[i] and b[i] have different sign bits
    __m256i sign_mask_ab = _mm256_cmpeq_epi8(sign_mask_a_sub_b, zero_vec);
    __m256i sign_mask2_a2_b2 = _mm256_cmpeq_epi8(sign_mask_a2_sub_b2, zero_vec);
    __m256i sign_mask_ad = _mm256_cmpeq_epi8(sign_mask_a_sub_d, zero_vec);
    __m256i sign_mask2_a2_d2 = _mm256_cmpeq_epi8(sign_mask_a2_sub_d2, zero_vec);
    __m256i sign_mask_cb = _mm256_cmpeq_epi8(sign_mask_c_sub_b, zero_vec);
    __m256i sign_mask2_c2_b2 = _mm256_cmpeq_epi8(sign_mask_c2_sub_b2, zero_vec);
    __m256i sign_mask_cd = _mm256_cmpeq_epi8(sign_mask_c_sub_d, zero_vec);
    __m256i sign_mask2_c2_d2 = _mm256_cmpeq_epi8(sign_mask_c2_sub_d2, zero_vec);

    __m256i corrected_ab = _mm256_blendv_epi8(b_negated, b_abs, sign_mask_ab);
    __m256i corrected_a2b2 = _mm256_blendv_epi8(b2_negated, b2_abs, sign_mask2_a2_b2);
    __m256i corrected_ad = _mm256_blendv_epi8(d_negated, d_abs, sign_mask_ad);
    __m256i corrected_a2d2 = _mm256_blendv_epi8(d2_negated, d2_abs, sign_mask2_a2_d2);
    __m256i corrected_cb = _mm256_blendv_epi8(b_negated, b_abs, sign_mask_cb);
    __m256i corrected_c2b2 = _mm256_blendv_epi8(b2_negated, b2_abs, sign_mask2_c2_b2);
    __m256i corrected_cd = _mm256_blendv_epi8(d_negated, d_abs, sign_mask_cd);
    __m256i corrected_c2d2 = _mm256_blendv_epi8(d2_negated, d2_abs, sign_mask2_c2_d2);

    // Multiply the absolute values of a_abs (unsigned 8-bit integers) and corrected_b (signed 8-bit integers)
    __m256i product_16x16_ab = _mm256_maddubs_epi16(a_abs, corrected_ab);
    __m256i product_16x16_ab2 = _mm256_maddubs_epi16(a2_abs, corrected_a2b2);
    __m256i product_16x16_ad = _mm256_maddubs_epi16(a_abs, corrected_ad);
    __m256i product_16x16_ad2 = _mm256_maddubs_epi16(a2_abs, corrected_a2d2);
    __m256i product_16x16_cb = _mm256_maddubs_epi16(c_abs, corrected_cb);
    __m256i product_16x16_cb2 = _mm256_maddubs_epi16(c2_abs, corrected_c2b2);
    __m256i product_16x16_cd = _mm256_maddubs_epi16(c_abs, corrected_cd);
    __m256i product_16x16_cd2 = _mm256_maddubs_epi16(c2_abs, corrected_c2d2);

    // Sign extend the 16-bit integers in vector to 32-bit integers
    __m256i ab_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ab, 0));
    __m256i ab2_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ab2, 0));
    __m256i ab_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ab, 1));
    __m256i ab2_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ab2, 1));
    __m256i ad_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ad, 0));
    __m256i ad2_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ad2, 0));
    __m256i ad_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ad, 1));
    __m256i ad2_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ad2, 1));
    __m256i cb_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cb, 0));
    __m256i cb2_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cb2, 0));
    __m256i cb_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cb, 1));
    __m256i cb2_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cb2, 1));
    __m256i cd_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cd, 0));
    __m256i cd2_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cd2, 0));
    __m256i cd_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cd, 1));
    __m256i cd2_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cd2, 1));

    // Element-wise add the 32-bit integer vectors
    // acc0 += a * b + a2 * b2, acc2 += a * d + a2 * d2, acc3 += c * b + c * b2, acc4 += c * d + c2 * d2
    acc0 = _mm256_add_epi32(acc0,
                            _mm256_add_epi32(_mm256_add_epi32(ab_ext1, ab2_ext1), _mm256_add_epi32(ab_ext2, ab2_ext2)));
    acc1 = _mm256_add_epi32(acc1,
                            _mm256_add_epi32(_mm256_add_epi32(ad_ext1, ad2_ext1), _mm256_add_epi32(ad_ext2, ad2_ext2)));
    acc2 = _mm256_add_epi32(acc2,
                            _mm256_add_epi32(_mm256_add_epi32(cb_ext1, cb2_ext1), _mm256_add_epi32(cb_ext2, cb2_ext2)));
    acc3 = _mm256_add_epi32(acc3,
                            _mm256_add_epi32(_mm256_add_epi32(cd_ext1, cd2_ext1), _mm256_add_epi32(cd_ext2, cd2_ext2)));
}

static inline void multiply_signed_int8_2x2_32epi(__m256i &a, __m256i &b, __m256i &c, __m256i &d, __m256i &acc0,
                                                  __m256i &acc1, __m256i &acc2, __m256i &acc3) {
    __m256i a_sign_mask = _mm256_cmpgt_epi8(zero_vec, a);  // set 0xFF if zero_vec[i] > a[i]
    __m256i b_sign_mask = _mm256_cmpgt_epi8(zero_vec, b);  // set 0xFF if zero_vec[i] > a[i]
    __m256i c_sign_mask = _mm256_cmpgt_epi8(zero_vec, c);
    __m256i d_sign_mask = _mm256_cmpgt_epi8(zero_vec, d);

    // Compute the two's complement of a, put it here for higher throughput with good instruction dep.
    __m256i b_abs = _mm256_abs_epi8(b);
    __m256i a_abs = _mm256_abs_epi8(a);
    __m256i d_abs = _mm256_abs_epi8(d);
    __m256i c_abs = _mm256_abs_epi8(c);
    __m256i b_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), b_abs);
    __m256i d_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), d_abs);

    // Manipulate the `sign` of B to represent the sign of the 16 bit result
    __m256i sign_mask_a_sub_b = _mm256_sub_epi8(a_sign_mask, b_sign_mask);
    __m256i sign_mask_a_sub_d = _mm256_sub_epi8(a_sign_mask, d_sign_mask);
    __m256i sign_mask_c_sub_b = _mm256_sub_epi8(c_sign_mask, b_sign_mask);
    __m256i sign_mask_c_sub_d = _mm256_sub_epi8(c_sign_mask, d_sign_mask);

    // sign_mask[i] if a[i] and b[i] have different sign bits
    __m256i sign_mask_ab = _mm256_cmpeq_epi8(sign_mask_a_sub_b, zero_vec);
    __m256i sign_mask_ad = _mm256_cmpeq_epi8(sign_mask_a_sub_d, zero_vec);
    __m256i sign_mask_cb = _mm256_cmpeq_epi8(sign_mask_c_sub_b, zero_vec);
    __m256i sign_mask_cd = _mm256_cmpeq_epi8(sign_mask_c_sub_d, zero_vec);

    __m256i corrected_ab = _mm256_blendv_epi8(b_negated, b_abs, sign_mask_ab);
    __m256i corrected_ad = _mm256_blendv_epi8(d_negated, d_abs, sign_mask_ad);
    __m256i corrected_cb = _mm256_blendv_epi8(b_negated, b_abs, sign_mask_cb);
    __m256i corrected_cd = _mm256_blendv_epi8(d_negated, d_abs, sign_mask_cd);

    // Multiply the absolute values of a_abs (unsigned 8-bit integers) and corrected_b (signed 8-bit integers)
    __m256i product_16x16_ab = _mm256_maddubs_epi16(a_abs, corrected_ab);
    __m256i product_16x16_ad = _mm256_maddubs_epi16(a_abs, corrected_ad);
    __m256i product_16x16_cb = _mm256_maddubs_epi16(c_abs, corrected_cb);
    __m256i product_16x16_cd = _mm256_maddubs_epi16(c_abs, corrected_cd);

    // Sign extend the 16-bit integers in vector to 32-bit integers
    __m256i ab_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ab, 0));
    __m256i ab_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ab, 1));
    __m256i ad_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ad, 0));
    __m256i ad_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ad, 1));
    __m256i cb_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cb, 0));
    __m256i cb_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cb, 1));
    __m256i cd_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cd, 0));
    __m256i cd_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cd, 1));

    // Element-wise add the 32-bit integer vectors
    // acc0 += a * b + a2 * b2, acc2 += a * d + a2 * d2, acc3 += c * b + c * b2, acc4 += c * d + c2 * d2
    acc0 = _mm256_add_epi32(acc0, _mm256_add_epi32(ab_ext1, ab_ext2));
    acc1 = _mm256_add_epi32(acc1, _mm256_add_epi32(ad_ext1, ad_ext2));
    acc2 = _mm256_add_epi32(acc2, _mm256_add_epi32(cb_ext1, cb_ext2));
    acc3 = _mm256_add_epi32(acc3, _mm256_add_epi32(cd_ext1, cd_ext2));
}

static inline void multiply_signed_int8_2x2_32epi_of(__m256i &a, __m256i &b, __m256i &c, __m256i &d, __m256i &acc0,
                                                     __m256i &acc1, __m256i &acc2, __m256i &acc3) {
    // Multiply the absolute values of a_abs (unsigned 8-bit integers) and corrected_b (signed 8-bit integers)
    __m256i product_16x16_ab = _mm256_maddubs_epi16(a, b);
    __m256i product_16x16_ad = _mm256_maddubs_epi16(a, d);
    __m256i product_16x16_cb = _mm256_maddubs_epi16(c, b);
    __m256i product_16x16_cd = _mm256_maddubs_epi16(c, d);

    // Sign extend the 16-bit integers in vector to 32-bit integers
    __m256i ab_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ab, 0));
    __m256i ab_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ab, 1));
    __m256i ad_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ad, 0));
    __m256i ad_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_ad, 1));
    __m256i cb_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cb, 0));
    __m256i cb_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cb, 1));
    __m256i cd_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cd, 0));
    __m256i cd_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16_cd, 1));

    // Element-wise add the 32-bit integer vectors
    // acc0 += a * b + a2 * b2, acc2 += a * d + a2 * d2, acc3 += c * b + c * b2, acc4 += c * d + c2 * d2
    acc0 = _mm256_add_epi32(acc0, _mm256_add_epi32(ab_ext1, ab_ext2));
    acc1 = _mm256_add_epi32(acc1, _mm256_add_epi32(ad_ext1, ad_ext2));
    acc2 = _mm256_add_epi32(acc2, _mm256_add_epi32(cb_ext1, cb_ext2));
    acc3 = _mm256_add_epi32(acc3, _mm256_add_epi32(cd_ext1, cd_ext2));
}

void *mat_mul_accelerator_int8_thread_func_2x2_32unroll(void *args) {
    int i, j, k;
    struct thread_args *thread_args = (struct thread_args *)args;
    const struct matmul_params *params = thread_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int32_t A_zp = A->qparams.zero_point, C_zp = C->qparams.zero_point;
    float A_sc = A->qparams.scale, B_sc = B->qparams.scale, C_sc = C->qparams.scale;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr, *data_C = C->int8_data_ptr;
    int start_i = thread_args->start_i, end_i = thread_args->end_i;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    assert(A->column % 32 == 0);

    for (i = start_i; i < end_i; i += 2) {
        if (i + 2 > end_i) {
            for (j = 0; j < C->column; j++) {
                int acc = 0;
                __m256i acc0_8x32 = _mm256_setzero_si256();
                for (k = 0; k < A->column; k += 32) {
                    __m256i aa = _mm256_loadu_si256((const __m256i_u *)&data_A[i * A->column + k]);
                    // assume B is transposed
                    __m256i bb = _mm256_loadu_si256((const __m256i_u *)&data_B[j * B->row + k]);
                    acc0_8x32 = _mm256_add_epi32(acc0_8x32, multiply_signed_int8_32epi(aa, bb));
                }
                int32_t *accptr = (int32_t *)&acc0_8x32;
                acc = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
                acc = (int32_t)std::round((float)acc * alpha + (float)(params->bias.int8_data_ptr[j]) * beta);
                acc -= C_zp;
                acc = MAX(acc, q_min);
                acc = MIN(acc, q_max);
                data_C[i * C->column + j] = (int8_t)acc;
            }
        } else {
            for (j = 0; j < C->column; j += 2) {
                // (i, j), (i, j+1), (i+1, j), (i+1, j+1)
                int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                __m256i acc0_8x32 = _mm256_setzero_si256(), acc1_8x32 = _mm256_setzero_si256(),
                        acc2_8x32 = _mm256_setzero_si256(), acc3_8x32 = _mm256_setzero_si256();
                for (k = 0; k < A->column; k += 32) {
                    __m256i aa = _mm256_loadu_si256((const __m256i_u *)&data_A[i * A->column + k]);
                    __m256i cc = _mm256_loadu_si256((const __m256i_u *)&data_A[(i + 1) * A->column + k]);
                    __m256i bb = _mm256_loadu_si256((const __m256i_u *)&data_B[j * B->row + k]);
                    __m256i dd = _mm256_loadu_si256((const __m256i_u *)&data_B[(j + 1) * B->row + k]);

                    multiply_signed_int8_2x2_32epi(aa, bb, cc, dd, acc0_8x32, acc1_8x32, acc2_8x32, acc3_8x32);
                }
                int32_t *accptr = (int32_t *)&acc0_8x32;
                acc0 = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
                accptr = (int32_t *)&acc1_8x32;
                acc1 = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
                accptr = (int32_t *)&acc2_8x32;
                acc2 = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
                accptr = (int32_t *)&acc3_8x32;
                acc3 = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];

                acc0 = (int32_t)std::round((float)acc0 * alpha + (float)(params->bias.int8_data_ptr[j]) * beta);
                acc1 = (int32_t)std::round((float)acc1 * alpha + (float)(params->bias.int8_data_ptr[j + 1]) * beta);
                acc2 = (int32_t)std::round((float)acc2 * alpha + (float)(params->bias.int8_data_ptr[j]) * beta);
                acc3 = (int32_t)std::round((float)acc3 * alpha + (float)(params->bias.int8_data_ptr[j + 1]) * beta);

                acc0 -= C_zp;
                acc1 -= C_zp;
                acc2 -= C_zp;
                acc3 -= C_zp;

                acc0 = MAX(acc0, q_min);
                acc1 = MAX(acc1, q_min);
                acc2 = MAX(acc2, q_min);
                acc3 = MAX(acc3, q_min);
                acc0 = MIN(acc0, q_max);
                acc1 = MIN(acc1, q_max);
                acc2 = MIN(acc2, q_max);
                acc3 = MIN(acc3, q_max);
                data_C[i * C->column + j] = (int8_t)acc0;
                data_C[i * C->column + j + 1] = (int8_t)acc1;
                data_C[(i + 1) * C->column + j] = (int8_t)acc2;
                data_C[(i + 1) * C->column + j + 1] = (int8_t)acc3;
            }
        }
    }
    return NULL;
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll(const struct matmul_params *params) {
    int j, num_thread = params->opt_params.num_thread;

    assert(params->A.column % 64 == 0);
    assert((params->C.column) % 2 == 0);

    if (num_thread > params->C.row) num_thread = params->C.row;

    pthread_t thread_pool[num_thread];
    struct thread_args threads_args[num_thread];

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_i = j * (params->C.row / num_thread);
        if (j == num_thread - 1)
            threads_args[j].end_i = params->C.row;
        else
            threads_args[j].end_i = (j + 1) * (params->C.row / num_thread);
        threads_args[j].blk_size = params->opt_params.blk_size;
        threads_args[j].params = params;
        pthread_create(&thread_pool[j], NULL, mat_mul_accelerator_int8_thread_func_2x2_32unroll, &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) {
        pthread_join(thread_pool[j], NULL);
    }
}

// Note: no expecting min/max clipping for this op, default to int8 range/ReLU if q_min == 0
void *mat_mul_accelerator_int8_fast_32unroll_over_column_thread_func(void *args) {
    int i, j, k;
    struct thread_args *thread_args = (struct thread_args *)args;
    const struct matmul_params *params = thread_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float A_sc = A->qparams.scale, B_sc = B->qparams.scale, C_sc = C->qparams.scale;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr, *data_C = C->int8_data_ptr;
    int start_i = thread_args->start_i, end_i = thread_args->end_i;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;
    float beta = params->beta;
    float alpha = params->alpha;

    for (i = 0; i < C->row; i++) {
        for (j = start_i; j < end_i; j += 4) {
            int acc = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0;
            __m256i acc0_8x32 = _mm256_setzero_si256();
            __m256i acc1_8x32 = _mm256_setzero_si256();
            __m256i acc2_8x32 = _mm256_setzero_si256();
            __m256i acc3_8x32 = _mm256_setzero_si256();
            __m256i *aa = (__m256i *)&data_A[i * A->column];
            __m256i *bb = (__m256i *)&data_B[j * B->row];
            __m256i *bb1 = (__m256i *)&data_B[(j + 1) * B->row];
            __m256i *bb2 = (__m256i *)&data_B[(j + 2) * B->row];
            __m256i *bb3 = (__m256i *)&data_B[(j + 3) * B->row];

            int blocks = A->column / 32;
            while (blocks) {
                // prefetch the next vars
                if (blocks > 1) {
                    _mm_prefetch(aa + 1, _MM_HINT_T0);
                    _mm_prefetch(bb + 1, _MM_HINT_T0);
                    _mm_prefetch(bb1 + 1, _MM_HINT_T0);
                    _mm_prefetch(bb2 + 1, _MM_HINT_T0);
                    _mm_prefetch(bb3 + 1, _MM_HINT_T0);
                }
                // compute
                acc0_8x32 = _mm256_add_epi32(acc0_8x32, multiply_signed_int8_32epi(*aa, *bb++));
                acc1_8x32 = _mm256_add_epi32(acc1_8x32, multiply_signed_int8_32epi(*aa, *bb1++));
                acc2_8x32 = _mm256_add_epi32(acc2_8x32, multiply_signed_int8_32epi(*aa, *bb2++));
                acc3_8x32 = _mm256_add_epi32(acc3_8x32, multiply_signed_int8_32epi(*aa++, *bb3++));
                blocks--;
            }
            assign_8int32((int32_t *)&acc0_8x32, acc);
            acc = (int32_t)std::round((float)(acc)*alpha + (float)(params->bias.int8_data_ptr[j]) * beta);
            assign_8int32((int32_t *)&acc1_8x32, acc1);
            acc1 = (int32_t)std::round((float)(acc1)*alpha + (float)(params->bias.int8_data_ptr[j + 1]) * beta);
            assign_8int32((int32_t *)&acc2_8x32, acc2);
            acc2 = (int32_t)std::round((float)(acc2)*alpha + (float)(params->bias.int8_data_ptr[j + 2]) * beta);
            assign_8int32((int32_t *)&acc3_8x32, acc3);
            acc3 = (int32_t)std::round((float)(acc3)*alpha + (float)(params->bias.int8_data_ptr[j + 3]) * beta);

            // ReLU
            if (q_min == 0) {
                acc = acc < 0 ? 0 : acc;
                acc1 = acc1 < 0 ? 0 : acc1;
                acc2 = acc2 < 0 ? 0 : acc2;
                acc3 = acc3 < 0 ? 0 : acc3;
            }

            data_C[i * C->column + j] = (int8_t)acc;
            data_C[i * C->column + j + 1] = (int8_t)acc1;
            data_C[i * C->column + j + 2] = (int8_t)acc2;
            data_C[i * C->column + j + 3] = (int8_t)acc3;
        }
    }
    return NULL;
}

void MatmulOperator::mat_mul_accelerator_int8_fast_32unroll_over_column(const struct matmul_params *params) {
    int j, num_thread = params->opt_params.num_thread;

    if (num_thread > params->C.column) num_thread = params->C.column;

    assert(params->A.column % 32 == 0);
    assert((params->C.column) % (num_thread * 8) == 0);

    pthread_t thread_pool[num_thread];
    struct thread_args threads_args[num_thread];

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_i = j * (params->C.column / num_thread);
        if (j == num_thread - 1)
            threads_args[j].end_i = params->C.column;
        else
            threads_args[j].end_i = (j + 1) * (params->C.column / num_thread);
        threads_args[j].blk_size = params->opt_params.blk_size;
        threads_args[j].params = params;
        pthread_create(&thread_pool[j], NULL, mat_mul_accelerator_int8_fast_32unroll_over_column_thread_func,
                       &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) {
        pthread_join(thread_pool[j], NULL);
    }
}

void *mat_mul_accelerator_int8_thread_func_2x2_32unroll_nobias(void *args) {
    int i, j, k;
    struct thread_args *thread_args = (struct thread_args *)args;
    const struct matmul_params *params = thread_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int32_t A_zp = A->qparams.zero_point, C_zp = C->qparams.zero_point;
    float A_sc = A->qparams.scale, B_sc = B->qparams.scale, C_sc = C->qparams.scale;
    float effective_scale = A_sc * B_sc / C_sc;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr, *data_C = C->int8_data_ptr;
    int start_i = thread_args->start_i, end_i = thread_args->end_i;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;

    for (i = start_i; i < end_i; i += 2) {
        if (i + 2 > end_i) {
            for (j = 0; j < C->column; j++) {
                int acc = 0;
                __m256i acc0_8x32 = _mm256_setzero_si256();
                int k_block = A->column / 32;
                int remains = A->column % 32;

                for (int kb = 0; kb < k_block; kb++) {
                    int k = 32 * kb;
                    __m256i aa = _mm256_loadu_si256((const __m256i_u *)&data_A[i * A->column + k]);
                    // assume B is transposed
                    __m256i bb = _mm256_loadu_si256((const __m256i_u *)&data_B[j * B->row + k]);
                    acc0_8x32 = _mm256_add_epi32(acc0_8x32, multiply_signed_int8_32epi(aa, bb));
                }
                int start_k = k_block * 32;
                for (int r = 0; r < remains; r++) {
                    acc += data_A[i * A->column + (start_k + r)] * data_B[j * B->row + (start_k + r)];
                }
                int32_t *accptr = (int32_t *)&acc0_8x32;
                acc += (accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7]);
                acc = (int32_t)std::round((float)acc * effective_scale);
                acc -= C_zp;
                acc = MAX(acc, q_min);
                acc = MIN(acc, q_max);
                data_C[i * C->column + j] = (int8_t)acc;
            }
        } else {
            assert(j % 2 == 0);
            for (j = 0; j < C->column; j += 2) {
                // (i, j), (i, j+1), (i+1, j), (i+1, j+1)
                int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                __m256i acc0_8x32 = _mm256_setzero_si256(), acc1_8x32 = _mm256_setzero_si256(),
                        acc2_8x32 = _mm256_setzero_si256(), acc3_8x32 = _mm256_setzero_si256();
                // Handle different sqlen
                int k_block = A->column / 32;
                int remains = A->column % 32;
                for (int kb = 0; kb < k_block; kb++) {
                    int k = 32 * kb;
                    __m256i aa = _mm256_loadu_si256((const __m256i_u *)&data_A[i * A->column + k]);
                    __m256i cc = _mm256_loadu_si256((const __m256i_u *)&data_A[(i + 1) * A->column + k]);
                    __m256i bb = _mm256_loadu_si256((const __m256i_u *)&data_B[j * B->row + k]);
                    __m256i dd = _mm256_loadu_si256((const __m256i_u *)&data_B[(j + 1) * B->row + k]);

                    // multiply_signed_int8_2x2_32epi_of(aa, bb, cc, dd, acc0_8x32, acc1_8x32, acc2_8x32, acc3_8x32);
                    multiply_signed_int8_2x2_32epi(aa, bb, cc, dd, acc0_8x32, acc1_8x32, acc2_8x32, acc3_8x32);
                }
                int start_k = k_block * 32;
                for (int r = 0; r < remains; r++) {
                    acc0 += data_A[i * A->column + (start_k + r)] * data_B[j * B->row + (start_k + r)];
                    acc1 += data_A[i * A->column + (start_k + r)] * data_B[(j + 1) * B->row + (start_k + r)];
                    acc2 += data_A[(i + 1) * A->column + (start_k + r)] * data_B[j * B->row + (start_k + r)];
                    acc3 += data_A[(i + 1) * A->column + (start_k + r)] * data_B[(j + 1) * B->row + (start_k + r)];
                }
                int32_t *accptr = (int32_t *)&acc0_8x32;
                acc0 += (accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7]);
                accptr = (int32_t *)&acc1_8x32;
                acc1 += (accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7]);
                accptr = (int32_t *)&acc2_8x32;
                acc2 += (accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7]);
                accptr = (int32_t *)&acc3_8x32;
                acc3 += (accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7]);

                acc0 = (int32_t)std::round((float)acc0 * effective_scale);
                acc1 = (int32_t)std::round((float)acc1 * effective_scale);
                acc2 = (int32_t)std::round((float)acc2 * effective_scale);
                acc3 = (int32_t)std::round((float)acc3 * effective_scale);

                acc0 -= C_zp;
                acc1 -= C_zp;
                acc2 -= C_zp;
                acc3 -= C_zp;

                acc0 = MAX(acc0, q_min);
                acc1 = MAX(acc1, q_min);
                acc2 = MAX(acc2, q_min);
                acc3 = MAX(acc3, q_min);
                acc0 = MIN(acc0, q_max);
                acc1 = MIN(acc1, q_max);
                acc2 = MIN(acc2, q_max);
                acc3 = MIN(acc3, q_max);
                data_C[i * C->column + j] = (int8_t)acc0;
                data_C[i * C->column + j + 1] = (int8_t)acc1;
                data_C[(i + 1) * C->column + j] = (int8_t)acc2;
                data_C[(i + 1) * C->column + j + 1] = (int8_t)acc3;
            }
        }
    }
    return NULL;
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias(const struct matmul_params *params) {
    int j, num_thread = params->opt_params.num_thread;

    assert((params->C.column) % 2 == 0);

    if (num_thread > params->C.row) num_thread = params->C.row;

    pthread_t thread_pool[num_thread];
    struct thread_args threads_args[num_thread];

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_i = j * (params->C.row / num_thread);
        if (j == num_thread - 1)
            threads_args[j].end_i = params->C.row;
        else
            threads_args[j].end_i = (j + 1) * (params->C.row / num_thread);
        threads_args[j].blk_size = params->opt_params.blk_size;
        threads_args[j].params = params;
        pthread_create(&thread_pool[j], NULL, mat_mul_accelerator_int8_thread_func_2x2_32unroll_nobias,
                       &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) {
        pthread_join(thread_pool[j], NULL);
    }
}

void *mat_mul_accelerator_int8_thread_func_2x2_32unroll_nobias_batch(void *args) {
    int i, j, k;
    struct thread_args *thread_args = (struct thread_args *)args;
    const struct matmul_params *params = thread_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int32_t A_zp = A->qparams.zero_point, C_zp = C->qparams.zero_point;
    float A_sc = A->qparams.scale, B_sc = B->qparams.scale, C_sc = C->qparams.scale;
    float effective_scale = A_sc * B_sc / C_sc;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr, *data_C = C->int8_data_ptr;
    int start_i = thread_args->start_i, end_i = thread_args->end_i;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;

    for (i = start_i; i < end_i; i++) {
        for (j = 0; j < C->column; j++) {
            int acc = 0;
            __m256i acc0_8x32 = _mm256_setzero_si256();
            int k_block = A->column / 32;
            int remains = A->column % 32;

            for (int kb = 0; kb < k_block; kb++) {
                int k = 32 * kb;
                __m256i aa = _mm256_loadu_si256((const __m256i_u *)&data_A[i * A->column + k]);
                // assume B is transposed
                __m256i bb = _mm256_loadu_si256((const __m256i_u *)&data_B[i * B->row * B->column + j * B->row + k]);
                acc0_8x32 = _mm256_add_epi32(acc0_8x32, multiply_signed_int8_32epi(aa, bb));
            }
            int start_k = k_block * 32;
            for (int r = 0; r < remains; r++) {
                acc +=
                    data_A[i * A->column + (start_k + r)] * data_B[i * B->row * B->column + j * B->row + (start_k + r)];
            }
            int32_t *accptr = (int32_t *)&acc0_8x32;
            acc += (accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7]);
            acc = (int32_t)std::round((float)acc * effective_scale);
            acc -= C_zp;
            acc = MAX(acc, q_min);
            acc = MIN(acc, q_max);
            data_C[i * C->column + j] = (int8_t)acc;
        }
    }
    return NULL;
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_batch(const struct matmul_params *params) {
    int j, num_thread = params->opt_params.num_thread;

    assert((params->C.column) % 2 == 0);

    if (num_thread > params->C.row) num_thread = params->C.row;

    pthread_t thread_pool[num_thread];
    struct thread_args threads_args[num_thread];

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_i = j * (params->C.row / num_thread);
        if (j == num_thread - 1)
            threads_args[j].end_i = params->C.row;
        else
            threads_args[j].end_i = (j + 1) * (params->C.row / num_thread);
        threads_args[j].blk_size = params->opt_params.blk_size;
        threads_args[j].params = params;
        pthread_create(&thread_pool[j], NULL, mat_mul_accelerator_int8_thread_func_2x2_32unroll_nobias_batch,
                       &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) {
        pthread_join(thread_pool[j], NULL);
    }
}

void *mat_mul_accelerator_int8_thread_func_2x2_32unroll_nobias_ofp32(void *args) {
    int i, j, k;
    struct thread_args *thread_args = (struct thread_args *)args;
    const struct matmul_params *params = thread_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int32_t A_zp = A->qparams.zero_point, C_zp = C->qparams.zero_point;
    float A_sc = A->qparams.scale, B_sc = B->qparams.scale, C_sc = C->qparams.scale;
    float effective_scale = A_sc * B_sc / C_sc;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr;
    float *data_C = C->data_ptr;
    int start_i = thread_args->start_i, end_i = thread_args->end_i;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;

    for (i = start_i; i < end_i; i += 2) {
        if (i + 2 > end_i) {
            for (j = 0; j < C->column; j++) {
                int acc = 0;
                __m256i acc0_8x32 = _mm256_setzero_si256();
                for (k = 0; k < A->column; k += 32) {
                    __m256i aa = _mm256_loadu_si256((const __m256i_u *)&data_A[i * A->column + k]);
                    // assume B is transposed
                    __m256i bb = _mm256_loadu_si256((const __m256i_u *)&data_B[j * B->row + k]);
                    acc0_8x32 = _mm256_add_epi32(acc0_8x32, multiply_signed_int8_32epi(aa, bb));
                }
                int32_t *accptr = (int32_t *)&acc0_8x32;
                acc = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
                data_C[i * C->column + j] = ((float)acc * effective_scale);
            }
        } else if (C->column % 2 != 0) {
            for (j = 0; j < C->column; j++) {
                int acc = 0, acc1 = 0;
                __m256i acc0_8x32 = _mm256_setzero_si256();
                __m256i acc1_8x32 = _mm256_setzero_si256();
                for (k = 0; k < A->column; k += 32) {
                    __m256i aa = _mm256_loadu_si256((const __m256i_u *)&data_A[i * A->column + k]);
                    __m256i aa2 = _mm256_loadu_si256((const __m256i_u *)&data_A[(i + 1) * A->column + k]);
                    // assume B is transposed
                    __m256i bb = _mm256_loadu_si256((const __m256i_u *)&data_B[j * B->row + k]);
                    acc0_8x32 = _mm256_add_epi32(acc0_8x32, multiply_signed_int8_32epi(aa, bb));
                    acc1_8x32 = _mm256_add_epi32(acc1_8x32, multiply_signed_int8_32epi(aa2, bb));
                }
                int32_t *accptr = (int32_t *)&acc0_8x32;
                acc = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
                accptr = (int32_t *)&acc1_8x32;
                acc1 = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];

                data_C[i * C->column + j] = ((float)acc * effective_scale);
                data_C[(i + 1) * C->column + j] = ((float)acc1 * effective_scale);
            }
        } else {
            for (j = 0; j < C->column; j += 2) {
                // (i, j), (i, j+1), (i+1, j), (i+1, j+1)
                int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                __m256i acc0_8x32 = _mm256_setzero_si256(), acc1_8x32 = _mm256_setzero_si256(),
                        acc2_8x32 = _mm256_setzero_si256(), acc3_8x32 = _mm256_setzero_si256();
                for (k = 0; k < A->column; k += 32) {
                    __m256i aa = _mm256_loadu_si256((const __m256i_u *)&data_A[i * A->column + k]);
                    __m256i cc = _mm256_loadu_si256((const __m256i_u *)&data_A[(i + 1) * A->column + k]);
                    __m256i bb = _mm256_loadu_si256((const __m256i_u *)&data_B[j * B->row + k]);
                    __m256i dd = _mm256_loadu_si256((const __m256i_u *)&data_B[(j + 1) * B->row + k]);

                    multiply_signed_int8_2x2_32epi(aa, bb, cc, dd, acc0_8x32, acc1_8x32, acc2_8x32, acc3_8x32);
                }
                int32_t *accptr = (int32_t *)&acc0_8x32;
                acc0 = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
                accptr = (int32_t *)&acc1_8x32;
                acc1 = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
                accptr = (int32_t *)&acc2_8x32;
                acc2 = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
                accptr = (int32_t *)&acc3_8x32;
                acc3 = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];

                data_C[i * C->column + j] = ((float)acc0 * effective_scale);
                data_C[i * C->column + j + 1] = ((float)acc1 * effective_scale);
                data_C[(i + 1) * C->column + j] = ((float)acc2 * effective_scale);
                data_C[(i + 1) * C->column + j + 1] = ((float)acc3 * effective_scale);
            }
        }
    }
    return NULL;
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32(const struct matmul_params *params) {
    int j, num_thread = params->opt_params.num_thread;

    assert(params->A.column % 32 == 0);

    if (num_thread > params->C.row) num_thread = params->C.row;

    pthread_t thread_pool[num_thread];
    struct thread_args threads_args[num_thread];

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_i = j * (params->C.row / num_thread);
        if (j == num_thread - 1)
            threads_args[j].end_i = params->C.row;
        else
            threads_args[j].end_i = (j + 1) * (params->C.row / num_thread);
        threads_args[j].blk_size = params->opt_params.blk_size;
        threads_args[j].params = params;
        pthread_create(&thread_pool[j], NULL, mat_mul_accelerator_int8_thread_func_2x2_32unroll_nobias_ofp32,
                       &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) {
        pthread_join(thread_pool[j], NULL);
    }
}

// C->row and A->row are the batch dim
void *mat_mul_accelerator_int8_thread_func_2x2_32unroll_nobias_ofp32_batch(void *args) {
    int i, j, k;
    struct thread_args *thread_args = (struct thread_args *)args;
    const struct matmul_params *params = thread_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int32_t A_zp = A->qparams.zero_point, C_zp = C->qparams.zero_point;
    float A_sc = A->qparams.scale, B_sc = B->qparams.scale, C_sc = C->qparams.scale;
    float effective_scale = A_sc * B_sc / C_sc;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr;
    float *data_C = C->data_ptr;
    int start_i = thread_args->start_i, end_i = thread_args->end_i;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;

    for (i = start_i; i < end_i; i++) {
        for (j = 0; j < C->column; j++) {
            int acc = 0;
            __m256i acc0_8x32 = _mm256_setzero_si256();
            for (k = 0; k < A->column; k += 32) {
                __m256i aa = _mm256_loadu_si256((const __m256i_u *)&data_A[i * A->column + k]);
                // assume B is transposed
                __m256i bb = _mm256_loadu_si256((const __m256i_u *)&data_B[i * B->row * B->column + j * B->row + k]);
                acc0_8x32 = _mm256_add_epi32(acc0_8x32, multiply_signed_int8_32epi(aa, bb));
            }
            int32_t *accptr = (int32_t *)&acc0_8x32;
            acc = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
            data_C[i * C->column + j] = ((float)acc * effective_scale);
        }
    }
    return NULL;
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_nobias_ofp32_batch(const struct matmul_params *params) {
    int j, num_thread = params->opt_params.num_thread;

    assert(params->A.column % 32 == 0);

    if (num_thread > params->C.row) num_thread = params->C.row;

    pthread_t thread_pool[num_thread];
    struct thread_args threads_args[num_thread];

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_i = j * (params->C.row / num_thread);
        if (j == num_thread - 1)
            threads_args[j].end_i = params->C.row;
        else
            threads_args[j].end_i = (j + 1) * (params->C.row / num_thread);
        threads_args[j].blk_size = params->opt_params.blk_size;
        threads_args[j].params = params;
        pthread_create(&thread_pool[j], NULL, mat_mul_accelerator_int8_thread_func_2x2_32unroll_nobias_ofp32_batch,
                       &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) {
        pthread_join(thread_pool[j], NULL);
    }
}

void *mat_mul_accelerator_int8_thread_func_2x2_32unroll_bfp32_ofp32(void *args) {
    int i, j, k;
    struct thread_args *thread_args = (struct thread_args *)args;
    const struct matmul_params *params = thread_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int32_t A_zp = A->qparams.zero_point, C_zp = C->qparams.zero_point;
    float A_sc = A->qparams.scale, B_sc = B->qparams.scale, C_sc = C->qparams.scale;
    float effective_scale = A_sc * B_sc / C_sc;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr;
    float *data_C = C->data_ptr;
    int start_i = thread_args->start_i, end_i = thread_args->end_i;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;

    for (i = start_i; i < end_i; i += 2) {
        if (i + 2 > end_i) {
            for (j = 0; j < C->column; j++) {
                int acc = 0;
                __m256i acc0_8x32 = _mm256_setzero_si256();
                for (k = 0; k < A->column; k += 32) {
                    __m256i aa = _mm256_loadu_si256((const __m256i_u *)&data_A[i * A->column + k]);
                    // assume B is transposed
                    __m256i bb = _mm256_loadu_si256((const __m256i_u *)&data_B[j * B->row + k]);
                    acc0_8x32 = _mm256_add_epi32(acc0_8x32, multiply_signed_int8_32epi(aa, bb));
                }
                int32_t *accptr = (int32_t *)&acc0_8x32;
                acc = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
                data_C[i * C->column + j] = ((float)acc * effective_scale) + params->bias.data_ptr[j];
            }
        } else {
            for (j = 0; j < C->column; j += 2) {
                // (i, j), (i, j+1), (i+1, j), (i+1, j+1)
                int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                __m256i acc0_8x32 = _mm256_setzero_si256(), acc1_8x32 = _mm256_setzero_si256(),
                        acc2_8x32 = _mm256_setzero_si256(), acc3_8x32 = _mm256_setzero_si256();
                for (k = 0; k < A->column; k += 32) {
                    __m256i aa = _mm256_loadu_si256((const __m256i_u *)&data_A[i * A->column + k]);
                    __m256i cc = _mm256_loadu_si256((const __m256i_u *)&data_A[(i + 1) * A->column + k]);
                    __m256i bb = _mm256_loadu_si256((const __m256i_u *)&data_B[j * B->row + k]);
                    __m256i dd = _mm256_loadu_si256((const __m256i_u *)&data_B[(j + 1) * B->row + k]);

                    multiply_signed_int8_2x2_32epi(aa, bb, cc, dd, acc0_8x32, acc1_8x32, acc2_8x32, acc3_8x32);
                }
                int32_t *accptr = (int32_t *)&acc0_8x32;
                acc0 = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
                accptr = (int32_t *)&acc1_8x32;
                acc1 = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
                accptr = (int32_t *)&acc2_8x32;
                acc2 = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];
                accptr = (int32_t *)&acc3_8x32;
                acc3 = accptr[0] + accptr[1] + accptr[2] + accptr[3] + accptr[4] + accptr[5] + accptr[6] + accptr[7];

                data_C[i * C->column + j] = ((float)acc0 * effective_scale) + params->bias.data_ptr[j];
                data_C[i * C->column + j + 1] = ((float)acc1 * effective_scale) + params->bias.data_ptr[j + 1];
                data_C[(i + 1) * C->column + j] = ((float)acc2 * effective_scale) + params->bias.data_ptr[j];
                data_C[(i + 1) * C->column + j + 1] = ((float)acc3 * effective_scale) + params->bias.data_ptr[j + 1];
            }
        }
    }
    return NULL;
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32(const struct matmul_params *params) {
    int j, num_thread = params->opt_params.num_thread;

    assert(params->A.column % 64 == 0);
    assert((params->C.column) % 2 == 0);

    pthread_t thread_pool[num_thread];
    struct thread_args threads_args[num_thread];

    if (num_thread > params->C.row) num_thread = params->C.row;

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_i = j * (params->C.row / num_thread);
        if (j == num_thread - 1)
            threads_args[j].end_i = params->C.row;
        else
            threads_args[j].end_i = (j + 1) * (params->C.row / num_thread);
        threads_args[j].blk_size = params->opt_params.blk_size;
        threads_args[j].params = params;
        pthread_create(&thread_pool[j], NULL, mat_mul_accelerator_int8_thread_func_2x2_32unroll_bfp32_ofp32,
                       &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) {
        pthread_join(thread_pool[j], NULL);
    }
}

// acc0 += a * b, acc1 += a * b1, acc2 += a * b2, acc3 += a * b3
inline void multiply_signed_int8_32epi_4unroll_test(__m256i &a, __m256i &b, __m256i &b1, __m256i &b2, __m256i &b3,
                                                    __m256i &acc0, __m256i &acc1, __m256i &acc2, __m256i &acc3) {
    __m256i a_sign_mask = _mm256_cmpgt_epi8(zero_vec, a);  // set 0xFF if zero_vec[i] > a[i]
    __m256i b_sign_mask = _mm256_cmpgt_epi8(zero_vec, b);  // set 0xFF if zero_vec[i] > a[i]

    // Compute the two's complement of a, put it here for higher throughput with good instruction dep.
    __m256i b_abs = _mm256_abs_epi8(b);
    __m256i a_abs = _mm256_abs_epi8(a);
    __m256i b_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), b_abs);

    // Manipulate the `sign` of B to represent the sign of the 16 bit result
    __m256i sign_mask_a_sub_b = _mm256_sub_epi8(a_sign_mask, b_sign_mask);
    __m256i sign_mask_a_sub_b1 = _mm256_sub_epi8(a_sign_mask, b1);
    __m256i sign_mask_a_sub_b2 = _mm256_sub_epi8(a_sign_mask, b2);
    __m256i sign_mask_a_sub_b3 = _mm256_sub_epi8(a_sign_mask, b3);
    // sign_mask[i] if a[i] and b[i] have different sign bits
    __m256i sign_mask = _mm256_cmpeq_epi8(sign_mask_a_sub_b, zero_vec);
    __m256i sign_mask1 = _mm256_cmpeq_epi8(sign_mask_a_sub_b1, zero_vec);
    __m256i sign_mask2 = _mm256_cmpeq_epi8(sign_mask_a_sub_b2, zero_vec);
    __m256i sign_mask3 = _mm256_cmpeq_epi8(sign_mask_a_sub_b3, zero_vec);

    __m256i corrected_b = _mm256_blendv_epi8(b_negated, b_abs, sign_mask);
    __m256i corrected_b1 = _mm256_blendv_epi8(b_negated, b_abs, sign_mask1);
    __m256i corrected_b2 = _mm256_blendv_epi8(b_negated, b_abs, sign_mask2);
    __m256i corrected_b3 = _mm256_blendv_epi8(b_negated, b_abs, sign_mask3);

    // Multiply the absolute values of a_abs (unsigned 8-bit integers) and corrected_b (signed 8-bit integers)
    __m256i product_16x16 = _mm256_maddubs_epi16(a_abs, corrected_b);
    __m256i product1_16x16 = _mm256_maddubs_epi16(a_abs, corrected_b1);
    __m256i product2_16x16 = _mm256_maddubs_epi16(a_abs, corrected_b2);
    __m256i product3_16x16 = _mm256_maddubs_epi16(a_abs, corrected_b3);

    // Sign extend the 16-bit integers in vector to 32-bit integers
    __m256i a_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16, 0));
    __m256i a_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16, 1));
    __m256i a1_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product1_16x16, 0));
    __m256i a1_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product1_16x16, 1));
    __m256i a2_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product2_16x16, 0));
    __m256i a2_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product2_16x16, 1));
    __m256i a3_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product3_16x16, 0));
    __m256i a3_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product3_16x16, 1));

    // Element-wise add the 32-bit integer vectors
    __m256i sum0 = _mm256_add_epi32(a_ext1, a_ext2);
    __m256i sum1 = _mm256_add_epi32(a1_ext1, a1_ext2);
    __m256i sum2 = _mm256_add_epi32(a2_ext1, a2_ext2);
    __m256i sum3 = _mm256_add_epi32(a3_ext1, a3_ext2);

    acc0 = _mm256_add_epi32(acc0, sum0);
    acc1 = _mm256_add_epi32(acc1, sum1);
    acc2 = _mm256_add_epi32(acc2, sum2);
    acc3 = _mm256_add_epi32(acc3, sum3);
}

// acc0 += a * b, acc1 += a * b1, acc2 += a * b2, acc3 += a * b3
void multiply_signed_int8_32epi_4unroll(__m256i &a, __m256i &b, __m256i &b1, __m256i &b2, __m256i &b3, __m256i &acc0,
                                        __m256i &acc1, __m256i &acc2, __m256i &acc3) {
    __m256i a_sign_mask = _mm256_cmpgt_epi8(zero_vec, a);    // set 0xFF if zero_vec[i] > a[i]
    __m256i b_sign_mask = _mm256_cmpgt_epi8(zero_vec, b);    // set 0xFF if zero_vec[i] > a[i]
    __m256i b1_sign_mask = _mm256_cmpgt_epi8(zero_vec, b1);  // set 0xFF if zero_vec[i] > a[i]
    __m256i b2_sign_mask = _mm256_cmpgt_epi8(zero_vec, b2);  // set 0xFF if zero_vec[i] > a[i]
    __m256i b3_sign_mask = _mm256_cmpgt_epi8(zero_vec, b3);  // set 0xFF if zero_vec[i] > a[i]

    // Compute the two's complement of a, put it here for higher throughput with good instruction dep.
    __m256i b_abs = _mm256_abs_epi8(b);
    __m256i b1_abs = _mm256_abs_epi8(b1);
    __m256i b2_abs = _mm256_abs_epi8(b2);
    __m256i b3_abs = _mm256_abs_epi8(b3);
    __m256i a_abs = _mm256_abs_epi8(a);
    __m256i b_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), b_abs);
    __m256i b1_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), b1_abs);
    __m256i b2_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), b2_abs);
    __m256i b3_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), b3_abs);

    // Manipulate the `sign` of B to represent the sign of the 16 bit result
    __m256i sign_mask_a_sub_b = _mm256_sub_epi8(a_sign_mask, b_sign_mask);
    __m256i sign_mask_a_sub_b1 = _mm256_sub_epi8(a_sign_mask, b1_sign_mask);
    __m256i sign_mask_a_sub_b2 = _mm256_sub_epi8(a_sign_mask, b2_sign_mask);
    __m256i sign_mask_a_sub_b3 = _mm256_sub_epi8(a_sign_mask, b3_sign_mask);
    // sign_mask[i] if a[i] and b[i] have different sign bits
    __m256i sign_mask = _mm256_cmpeq_epi8(sign_mask_a_sub_b, zero_vec);
    __m256i sign_mask1 = _mm256_cmpeq_epi8(sign_mask_a_sub_b1, zero_vec);
    __m256i sign_mask2 = _mm256_cmpeq_epi8(sign_mask_a_sub_b2, zero_vec);
    __m256i sign_mask3 = _mm256_cmpeq_epi8(sign_mask_a_sub_b3, zero_vec);

    __m256i corrected_b = _mm256_blendv_epi8(b_negated, b_abs, sign_mask);
    __m256i corrected_b1 = _mm256_blendv_epi8(b1_negated, b1_abs, sign_mask1);
    __m256i corrected_b2 = _mm256_blendv_epi8(b2_negated, b2_abs, sign_mask2);
    __m256i corrected_b3 = _mm256_blendv_epi8(b3_negated, b3_abs, sign_mask3);

    // Multiply the absolute values of a_abs (unsigned 8-bit integers) and corrected_b (signed 8-bit integers)
    __m256i product_16x16 = _mm256_maddubs_epi16(a_abs, corrected_b);
    __m256i product1_16x16 = _mm256_maddubs_epi16(a_abs, corrected_b1);
    __m256i product2_16x16 = _mm256_maddubs_epi16(a_abs, corrected_b2);
    __m256i product3_16x16 = _mm256_maddubs_epi16(a_abs, corrected_b3);

    // Sign extend the 16-bit integers in vector to 32-bit integers
    __m256i a_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16, 0));
    __m256i a_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16, 1));
    __m256i a1_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product1_16x16, 0));
    __m256i a1_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product1_16x16, 1));
    __m256i a2_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product2_16x16, 0));
    __m256i a2_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product2_16x16, 1));
    __m256i a3_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product3_16x16, 0));
    __m256i a3_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product3_16x16, 1));

    // Element-wise add the 32-bit integer vectors
    __m256i sum0 = _mm256_add_epi32(a_ext1, a_ext2);
    __m256i sum1 = _mm256_add_epi32(a1_ext1, a1_ext2);
    __m256i sum2 = _mm256_add_epi32(a2_ext1, a2_ext2);
    __m256i sum3 = _mm256_add_epi32(a3_ext1, a3_ext2);

    acc0 = _mm256_add_epi32(acc0, sum0);
    acc1 = _mm256_add_epi32(acc1, sum1);
    acc2 = _mm256_add_epi32(acc2, sum2);
    acc3 = _mm256_add_epi32(acc3, sum3);
}

// acc0 += a * b, acc1 += a * b1, acc2 += a * b2, acc3 += a * b3 (a,b are 128bit : int8x16)
void multiply_signed_int8_16epi_4unroll(__m128i &a, __m128i &b, __m128i &b1, __m128i &b2, __m128i &b3, __m256i &acc0,
                                        __m256i &acc1, __m256i &acc2, __m256i &acc3) {
    // Expand int8 to int16
    __m256i a_ex = _mm256_cvtepi8_epi16(a);
    __m256i b_ex = _mm256_cvtepi8_epi16(b);
    __m256i b1_ex = _mm256_cvtepi8_epi16(b1);
    __m256i b2_ex = _mm256_cvtepi8_epi16(b2);
    __m256i b3_ex = _mm256_cvtepi8_epi16(b3);

    // Multiply the absolute values of a_abs (unsigned 8-bit integers) and corrected_b (signed 8-bit integers)
    __m256i product_16x16 = a_ex * b_ex;
    __m256i product1_16x16 = a_ex * b1_ex;
    __m256i product2_16x16 = a_ex * b2_ex;
    __m256i product3_16x16 = a_ex * b3_ex;

    // Sign extend the 16-bit integers in vector to 32-bit integers
    __m256i a_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16, 0));
    __m256i a_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16, 1));
    __m256i a1_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product1_16x16, 0));
    __m256i a1_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product1_16x16, 1));
    __m256i a2_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product2_16x16, 0));
    __m256i a2_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product2_16x16, 1));
    __m256i a3_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product3_16x16, 0));
    __m256i a3_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product3_16x16, 1));

    // Element-wise add the 32-bit integer vectors
    __m256i sum0 = _mm256_add_epi32(a_ext1, a_ext2);
    __m256i sum1 = _mm256_add_epi32(a1_ext1, a1_ext2);
    __m256i sum2 = _mm256_add_epi32(a2_ext1, a2_ext2);
    __m256i sum3 = _mm256_add_epi32(a3_ext1, a3_ext2);

    acc0 = _mm256_add_epi32(acc0, sum0);
    acc1 = _mm256_add_epi32(acc1, sum1);
    acc2 = _mm256_add_epi32(acc2, sum2);
    acc3 = _mm256_add_epi32(acc3, sum3);
}

inline void multiply_signed_int8_32epi_2unroll(__m256i &a, __m256i &b, __m256i &b1, __m256i &acc0, __m256i &acc1) {
    __m256i a_sign_mask = _mm256_cmpgt_epi8(zero_vec, a);    // set 0xFF if zero_vec[i] > a[i]
    __m256i b_sign_mask = _mm256_cmpgt_epi8(zero_vec, b);    // set 0xFF if zero_vec[i] > a[i]
    __m256i b1_sign_mask = _mm256_cmpgt_epi8(zero_vec, b1);  // set 0xFF if zero_vec[i] > a[i]

    // Compute the two's complement of a, put it here for higher throughput with good instruction dep.
    __m256i b_abs = _mm256_abs_epi8(b);
    __m256i b1_abs = _mm256_abs_epi8(b1);
    __m256i a_abs = _mm256_abs_epi8(a);
    __m256i b_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), b_abs);
    __m256i b1_negated = _mm256_sub_epi8(_mm256_set1_epi8(0), b1_abs);

    // Manipulate the `sign` of B to represent the sign of the 16 bit result
    __m256i sign_mask_a_sub_b = _mm256_sub_epi8(a_sign_mask, b_sign_mask);
    __m256i sign_mask_a_sub_b1 = _mm256_sub_epi8(a_sign_mask, b1_sign_mask);
    // sign_mask[i] if a[i] and b[i] have different sign bits
    __m256i sign_mask = _mm256_cmpeq_epi8(sign_mask_a_sub_b, zero_vec);
    __m256i sign_mask1 = _mm256_cmpeq_epi8(sign_mask_a_sub_b1, zero_vec);

    __m256i corrected_b = _mm256_blendv_epi8(b_negated, b_abs, sign_mask);
    __m256i corrected_b1 = _mm256_blendv_epi8(b1_negated, b1_abs, sign_mask1);

    // Multiply the absolute values of a_abs (unsigned 8-bit integers) and corrected_b (signed 8-bit integers)
    __m256i product_16x16 = _mm256_maddubs_epi16(a_abs, corrected_b);
    __m256i product1_16x16 = _mm256_maddubs_epi16(a_abs, corrected_b1);

    // Sign extend the 16-bit integers in vector to 32-bit integers
    __m256i a_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16, 0));
    __m256i a_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product_16x16, 1));
    __m256i a1_ext1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product1_16x16, 0));
    __m256i a1_ext2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(product1_16x16, 1));

    // Element-wise add the 32-bit integer vectors
    __m256i sum0 = _mm256_add_epi32(a_ext1, a_ext2);
    __m256i sum1 = _mm256_add_epi32(a1_ext1, a1_ext2);

    acc0 = _mm256_add_epi32(acc0, sum0);
    acc1 = _mm256_add_epi32(acc1, sum1);
}

void *mat_mul_accelerator_int8_thread_func_2x2_32unroll_bfp32_ofp32_over_column(void *args) {
    int i, j, k;
    struct thread_args *thread_args = (struct thread_args *)args;
    const struct matmul_params *params = thread_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    float A_sc = A->qparams.scale, B_sc = B->qparams.scale, C_sc = C->qparams.scale;
    float effective_scale = A_sc * B_sc / C_sc;
    int8_t *data_A = A->int8_data_ptr, *data_B = B->int8_data_ptr;
    float *data_C = C->data_ptr;
    int start_i = thread_args->start_i, end_i = thread_args->end_i;
    const int8_t q_min = C->qparams.q_min, q_max = C->qparams.q_max;

    for (i = 0; i < C->row; i++) {
        for (j = start_i; j < end_i; j += 4) {
            int acc = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            __m256i acc0_8x32 = _mm256_setzero_si256();
            __m256i acc1_8x32 = _mm256_setzero_si256();
            __m256i acc2_8x32 = _mm256_setzero_si256();
            __m256i acc3_8x32 = _mm256_setzero_si256();
            __m256i *aa_ptr = (__m256i *)&data_A[i * A->column];
            __m256i *bb_ptr = (__m256i *)&data_B[j * B->row];
            __m256i *bb1_ptr = (__m256i *)&data_B[(j + 1) * B->row];
            __m256i *bb2_ptr = (__m256i *)&data_B[(j + 2) * B->row];
            __m256i *bb3_ptr = (__m256i *)&data_B[(j + 3) * B->row];
            // TODO: precompute some masks to save some computation?
            int blocks = A->column / 32;
            while (blocks) {
                multiply_signed_int8_32epi_4unroll(*aa_ptr++, *bb_ptr++, *bb1_ptr++, *bb2_ptr++, *bb3_ptr++, acc0_8x32,
                                                   acc1_8x32, acc2_8x32, acc3_8x32);
                blocks--;
            }

            assign_8int32((int32_t *)&acc0_8x32, acc);
            assign_8int32((int32_t *)&acc1_8x32, acc1);
            assign_8int32((int32_t *)&acc2_8x32, acc2);
            assign_8int32((int32_t *)&acc3_8x32, acc3);
            data_C[i * C->column + j] = ((float)acc * effective_scale) + params->bias.data_ptr[j];
            data_C[i * C->column + j + 1] = ((float)acc1 * effective_scale) + params->bias.data_ptr[j + 1];
            data_C[i * C->column + j + 2] = ((float)acc2 * effective_scale) + params->bias.data_ptr[j + 2];
            data_C[i * C->column + j + 3] = ((float)acc3 * effective_scale) + params->bias.data_ptr[j + 3];
        }
    }
    return NULL;
}

void MatmulOperator::mat_mul_accelerator_int8_fast_2x2_32unroll_bfp32_ofp32_over_column(
    const struct matmul_params *params) {
    int j, num_thread = params->opt_params.num_thread;

    if (num_thread > params->C.column) num_thread = params->C.column;

    assert(params->A.column % 32 == 0);
    assert((params->C.column) % (num_thread * 4) == 0);

    pthread_t thread_pool[num_thread];
    struct thread_args threads_args[num_thread];

    // Thread creation
    for (j = 0; j < num_thread; j++) {
        threads_args[j].start_i = j * (params->C.column / num_thread);
        if (j == num_thread - 1)
            threads_args[j].end_i = params->C.column;
        else
            threads_args[j].end_i = (j + 1) * (params->C.column / num_thread);
        threads_args[j].blk_size = params->opt_params.blk_size;
        threads_args[j].params = params;
        pthread_create(&thread_pool[j], NULL, mat_mul_accelerator_int8_thread_func_2x2_32unroll_bfp32_ofp32_over_column,
                       &threads_args[j]);
    }
    // Join threads
    for (j = 0; j < num_thread; j++) {
        pthread_join(thread_pool[j], NULL);
    }
}

}  // namespace matmul
